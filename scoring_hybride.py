from sentence_transformers import SentenceTransformer
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient
from claude import ClaudeLLM
from typing import List, Dict, Tuple
from datetime import datetime
import json
import re  # AJOUT: pour le calcul de correspondance par mots-clés
from collections import Counter  # AJOUT: pour le scoring par mots-clés
import numpy as np  # AJOUT: pour les calculs de scoring


class ConversationMemory:
    """Gère la mémoire de conversation du chatbot"""

    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.conversation_history: List[Dict] = []
        self.session_start = datetime.now()

    def add_exchange(self, question: str, answer: str, context: str = ""):
        """Ajoute un échange question-réponse à l'historique"""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "context": context
        }

        self.conversation_history.append(exchange)

        # Limite la taille de l'historique
        if len(self.conversation_history) > self.max_history:
            self.conversation_history.pop(0)

    def get_recent_context(self, num_exchanges: int = 3) -> str:
        """Récupère le contexte des derniers échanges"""
        if not self.conversation_history:
            return ""

        recent_exchanges = self.conversation_history[-num_exchanges:]
        context_parts = []

        for exchange in recent_exchanges:
            context_parts.append(f"Q: {exchange['question']}")
            context_parts.append(f"R: {exchange['answer']}")

        return "\n".join(context_parts)

    # AJOUT: Nouvelle méthode pour extraire les mots-clés du contexte conversationnel
    def extract_context_keywords(self, num_exchanges: int = 5) -> List[str]:
        """Extrait les mots-clés importants de l'historique récent"""
        if not self.conversation_history:
            return []

        recent_exchanges = self.conversation_history[-num_exchanges:]
        text = " ".join([f"{ex['question']} {ex['answer']}" for ex in recent_exchanges])

        # Extraire les mots significatifs (plus de 3 caractères, pas de stop words basiques)
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', text.lower())
        stop_words = {'dans', 'avec', 'pour', 'mais', 'que', 'qui', 'quoi', 'comment', 'pourquoi', 'cette', 'cette'}
        keywords = [word for word in words if word not in stop_words]

        # Retourner les mots les plus fréquents
        word_counts = Counter(keywords)
        return [word for word, count in word_counts.most_common(10)]

    def get_full_history(self) -> List[Dict]:
        """Retourne l'historique complet"""
        return self.conversation_history.copy()

    def clear_history(self):
        """Efface l'historique de conversation"""
        self.conversation_history.clear()
        self.session_start = datetime.now()

    def save_to_file(self, filename: str):
        """Sauvegarde l'historique dans un fichier"""
        data = {
            "session_start": self.session_start.isoformat(),
            "conversation_history": self.conversation_history
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filename: str):
        """Charge l'historique depuis un fichier"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.conversation_history = data.get("conversation_history", [])
                self.session_start = datetime.fromisoformat(data.get("session_start", datetime.now().isoformat()))
        except FileNotFoundError:
            print(f"Fichier {filename} non trouvé. Démarrage avec un historique vide.")


class TVATunisiaBot:
    """Chatbot spécialisé en TVA tunisienne avec mémoire de conversation et scoring hybride"""

    def __init__(self, qdrant_host: str = "localhost", qdrant_port: int = 6333, max_history: int = 10):
        # Set up the embeddings
        self.embeddings = SentenceTransformer('BAAI/bge-small-en-v1.5')

        # Initialize conversation memory
        self.memory = ConversationMemory(max_history=max_history)

        # Set up Qdrant client
        self.qdrant_client = QdrantClient(qdrant_host, port=qdrant_port)

        # Set up vector store
        self.vectorstore = Qdrant(
            client=self.qdrant_client,
            collection_name="document_collection",
            embeddings=self._embedding_function,
            vector_name="content"
        )

        # MODIFICATION: Augmentation du nombre de documents récupérés pour le scoring hybride
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 15})  # Était 5, maintenant 15

        # Initialize Claude LLM
        self.claude_llm = ClaudeLLM()

        # AJOUT: Configuration du scoring hybride
        self.hybrid_weights = {
            'semantic': 0.6,  # Poids pour la similarité sémantique
            'keyword': 0.3,  # Poids pour la correspondance par mots-clés
            'context': 0.1  # Poids pour la pertinence contextuelle
        }

        # Template avec contexte de conversation
        self.template = """Notre sujet est la TVA en Tunisie.
Tu es un assistant pour les questions concernant la TVA tunisienne.

HISTORIQUE DE CONVERSATION:
{conversation_history}

CONTEXTE :
{document_context}

INSTRUCTIONS:
notre sujet est la tva en tunisie.
tu est un assistant pour question-answering tasks 
utilise le context que je t'ai donner et un peut de tes connaissance pour repondre au question de tva en tunisie 
si les question sort du sujet de la tva tu doit repondre quand meme mais apres avoir repondu tu doit avec gentillesse rappeler qu'il ne doit pas sortir de sujet.
quand vous repondez il ne faut pas mentionner que vous avez un contexte
Si tu fais référence à des éléments de la conversation précédente, fais-le naturellement

QUESTION ACTUELLE: {question}

RÉPONSE:"""

    def _embedding_function(self, text):
        """Fonction d'embedding pour Qdrant"""
        if isinstance(text, str):
            text = [text]
        return self.embeddings.encode(text)[0].tolist()

    # AJOUT: Méthode pour calculer le score de correspondance par mots-clés
    def _calculate_keyword_score(self, query: str, document_text: str) -> float:
        """Calcule le score de correspondance par mots-clés entre la requête et le document"""
        # Normaliser les textes
        query_words = set(re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', query.lower()))
        doc_words = set(re.findall(r'\b[a-zA-ZÀ-ÿ]{3,}\b', document_text.lower()))

        if not query_words:
            return 0.0

        # Calculer l'intersection
        common_words = query_words.intersection(doc_words)

        # Score basé sur le ratio de mots communs
        keyword_score = len(common_words) / len(query_words)

        # Bonus pour les mots exacts multiples
        if len(common_words) > 1:
            keyword_score *= 1.2

        return min(keyword_score, 1.0)

    # AJOUT: Méthode pour calculer le score contextuel
    def _calculate_context_score(self, document_text: str, context_keywords: List[str]) -> float:
        """Calcule le score de pertinence contextuelle basé sur l'historique"""
        if not context_keywords:
            return 0.0

        doc_text_lower = document_text.lower()
        matches = sum(1 for keyword in context_keywords if keyword in doc_text_lower)

        return matches / len(context_keywords)

    # AJOUT: Méthode principale de scoring hybride
    def _apply_hybrid_scoring(self, docs, query: str) -> List:
        """Applique le scoring hybride aux documents récupérés"""
        context_keywords = self.memory.extract_context_keywords()
        scored_docs = []

        for doc in docs:
            # Score sémantique (fourni par Qdrant via similarity_search_with_score)
            semantic_score = getattr(doc, 'metadata', {}).get('score', 0.8)  # Score par défaut si non disponible

            # Score par mots-clés
            keyword_score = self._calculate_keyword_score(query, doc.page_content)

            # Score contextuel
            context_score = self._calculate_context_score(doc.page_content, context_keywords)

            # Calcul du score hybride pondéré
            hybrid_score = (
                    self.hybrid_weights['semantic'] * semantic_score +
                    self.hybrid_weights['keyword'] * keyword_score +
                    self.hybrid_weights['context'] * context_score
            )

            # Stocker le score dans les métadonnées
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['hybrid_score'] = hybrid_score
            doc.metadata['semantic_score'] = semantic_score
            doc.metadata['keyword_score'] = keyword_score
            doc.metadata['context_score'] = context_score

            scored_docs.append(doc)

        # Trier par score hybride décroissant
        scored_docs.sort(key=lambda x: x.metadata.get('hybrid_score', 0), reverse=True)

        return scored_docs

    # AJOUT: Méthode pour sélectionner les meilleurs chunks avec diversité
    def _select_diverse_chunks(self, docs, max_chunks: int = 5, similarity_threshold: float = 0.85) -> List:
        """Sélectionne des chunks diversifiés pour éviter la redondance"""
        if not docs:
            return []

        selected = [docs[0]]  # Prendre le meilleur document

        for doc in docs[1:]:
            if len(selected) >= max_chunks:
                break

            # Vérifier la similarité avec les documents déjà sélectionnés
            is_diverse = True
            doc_embedding = self.embeddings.encode([doc.page_content])[0]

            for selected_doc in selected:
                selected_embedding = self.embeddings.encode([selected_doc.page_content])[0]

                # Calculer la similarité cosinus
                similarity = np.dot(doc_embedding, selected_embedding) / (
                        np.linalg.norm(doc_embedding) * np.linalg.norm(selected_embedding)
                )

                if similarity > similarity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(doc)

        return selected

    # MODIFICATION: Méthode get_response modifiée pour utiliser le scoring hybride
    def get_response(self, question: str) -> Tuple[str, str]:
        """Génère une réponse en utilisant le contexte, l'historique et le scoring hybride"""
        # Récupérer les documents pertinents (plus de documents pour le scoring)
        docs = self.retriever.get_relevant_documents(question)

        # AJOUT: Appliquer le scoring hybride
        hybrid_scored_docs = self._apply_hybrid_scoring(docs, question)

        # AJOUT: Sélectionner des chunks diversifiés
        final_docs = self._select_diverse_chunks(hybrid_scored_docs, max_chunks=5)

        # Créer le contexte des documents avec informations de scoring (optionnel pour debug)
        document_context_parts = []
        for i, doc in enumerate(final_docs):
            # Ajouter le contenu du document
            document_context_parts.append(doc.page_content)

            # AJOUT: Optionnel - afficher les scores pour debug (commenté par défaut)
            # hybrid_score = doc.metadata.get('hybrid_score', 0)
            # print(f"Doc {i+1} - Score hybride: {hybrid_score:.3f}")

        document_context = "\n".join(document_context_parts)

        # Récupérer l'historique de conversation
        conversation_history = self.memory.get_recent_context(num_exchanges=3)

        # Créer le prompt avec contexte et historique
        prompt = self.template.format(
            conversation_history=conversation_history,
            document_context=document_context,
            question=question
        )

        # Générer la réponse
        response = self.claude_llm.generate_response(prompt)
        answer = response[0].text if hasattr(response[0], 'text') else str(response[0])

        # Ajouter l'échange à la mémoire
        self.memory.add_exchange(question, answer, document_context)

        return document_context, answer

    # AJOUT: Méthode pour ajuster les poids du scoring hybride
    def set_hybrid_weights(self, semantic: float = 0.6, keyword: float = 0.3, context: float = 0.1):
        """Permet d'ajuster les poids du scoring hybride"""
        total = semantic + keyword + context
        if abs(total - 1.0) > 0.01:  # Tolérance pour les erreurs de float
            print(f"Attention: La somme des poids ({total}) n'est pas égale à 1.0")

        self.hybrid_weights = {
            'semantic': semantic,
            'keyword': keyword,
            'context': context
        }
        print(f"Poids mis à jour: Sémantique={semantic}, Mots-clés={keyword}, Contexte={context}")

    # AJOUT: Méthode pour afficher les statistiques de scoring
    def show_scoring_info(self):
        """Affiche les informations sur la configuration du scoring hybride"""
        print("\n=== CONFIGURATION SCORING HYBRIDE ===")
        print(f"Poids sémantique: {self.hybrid_weights['semantic']}")
        print(f"Poids mots-clés: {self.hybrid_weights['keyword']}")
        print(f"Poids contexte: {self.hybrid_weights['context']}")
        print(f"Nombre de documents récupérés: {self.retriever.search_kwargs['k']}")
        print("=" * 38)

    def show_history(self):
        """Affiche l'historique de conversation"""
        history = self.memory.get_full_history()
        if not history:
            print("Aucun historique de conversation.")
            return

        print("\n=== HISTORIQUE DE CONVERSATION ===")
        for i, exchange in enumerate(history, 1):
            print(f"\n[{i}] {exchange['timestamp']}")
            print(f"Q: {exchange['question']}")
            print(f"R: {exchange['answer'][:200]}..." if len(exchange['answer']) > 200 else f"R: {exchange['answer']}")
        print("=" * 35)

    def clear_memory(self):
        """Efface la mémoire de conversation"""
        self.memory.clear_history()
        print("Mémoire de conversation effacée.")

    def save_conversation(self, filename: str):
        """Sauvegarde la conversation"""
        self.memory.save_to_file(filename)
        print(f"Conversation sauvegardée dans {filename}")

    def load_conversation(self, filename: str):
        """Charge une conversation sauvegardée"""
        self.memory.load_from_file(filename)
        print(f"Conversation chargée depuis {filename}")


def main():
    """Fonction principale avec interface utilisateur améliorée"""
    bot = TVATunisiaBot()

    print("=== Chatbot TVA Tunisie avec Scoring Hybride ===")  # MODIFICATION: Titre mis à jour
    print("Commandes spéciales:")
    print("- 'quit' ou 'exit': quitter")
    print("- 'history': voir l'historique")
    print("- 'clear': effacer la mémoire")
    print("- 'save <nom_fichier>': sauvegarder la conversation")
    print("- 'load <nom_fichier>': charger une conversation")
    print("- 'scoring': voir la configuration du scoring hybride")  # AJOUT: Nouvelle commande
    print("- 'weights <sem> <key> <ctx>': ajuster les poids (ex: weights 0.5 0.4 0.1)")  # AJOUT: Nouvelle commande
    print("=" * 50)  # MODIFICATION: Ligne plus longue

    while True:
        question = input("\n🤖 Votre question: ").strip()

        if question.lower() in ['quit', 'exit']:
            print("Au revoir!")
            break

        elif question.lower() == 'history':
            bot.show_history()
            continue

        elif question.lower() == 'clear':
            bot.clear_memory()
            continue

        # AJOUT: Nouvelle commande pour voir la configuration du scoring
        elif question.lower() == 'scoring':
            bot.show_scoring_info()
            continue

        # AJOUT: Nouvelle commande pour ajuster les poids
        elif question.lower().startswith('weights '):
            try:
                parts = question.split()
                if len(parts) == 4:
                    sem, key, ctx = float(parts[1]), float(parts[2]), float(parts[3])
                    bot.set_hybrid_weights(sem, key, ctx)
                else:
                    print("Format: weights <sémantique> <mots-clés> <contexte>")
                    print("Exemple: weights 0.5 0.4 0.1")
            except ValueError:
                print("Erreur: veuillez utiliser des nombres décimaux")
            continue

        elif question.lower().startswith('save '):
            filename = question[5:].strip()
            if filename:
                bot.save_conversation(filename)
            else:
                print("Veuillez spécifier un nom de fichier: save <nom_fichier>")
            continue

        elif question.lower().startswith('load '):
            filename = question[5:].strip()
            if filename:
                bot.load_conversation(filename)
            else:
                print("Veuillez spécifier un nom de fichier: load <nom_fichier>")
            continue

        elif not question:
            continue

        try:
            context, response = bot.get_response(question)
            print(f"\n💬 Réponse: {response}")

        except Exception as e:
            print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    main()