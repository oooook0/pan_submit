from transformers import AutoTokenizer, AutoModel
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import variation
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import re

class ContextualRelevanceFeatures:
    def __init__(self):
        # Initialize BERT model for semantic analysis
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model = AutoModel.from_pretrained('bert-base-uncased')
        # Common discourse markers
        self.discourse_markers = {
            'causal': ['because', 'therefore', 'thus', 'hence', 'consequently', 'so'],
            'contrast': ['however', 'but', 'although', 'nevertheless', 'yet', 'despite'],
            'addition': ['moreover', 'furthermore', 'additionally', 'also', 'besides'],
            'temporal': ['then', 'next', 'afterwards', 'subsequently', 'finally'],
            'exemplification': ['for example', 'for instance', 'such as', 'specifically']
        }
        
    def extract_entity_relationships(self, doc):
        """Extract entities and their relationships"""        
        # Entity extraction
        entities = defaultdict(list)
        for ent in doc.ents:
            entities[ent.label_].append(ent.text)
            
        # Extract entity relationships
        relationships = []
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                    head = token.head.text
                    dependent = token.text
                    relationships.append((head, token.dep_, dependent))
                    
        return {
            'entity_diversity': len(entities),
            'relationship_count': len(relationships),
            'relationship_types': len(set(rel[1] for rel in relationships))
        }
    
    def analyze_anaphora(self, doc):
        """Analyze anaphora resolution patterns"""
        
        # Track pronouns and their potential referents
        pronouns = []
        referents = []
        
        for token in doc:
            if token.pos_ == 'PRON':
                pronouns.append(token)
            elif token.pos_ in ['NOUN', 'PROPN']:
                referents.append(token)
        
        # Calculate basic statistics
        return {
            'pronoun_density': len(pronouns) / len(doc),
            'pronoun_to_referent_ratio': len(pronouns) / (len(referents) + 1e-6),
            'unique_pronouns': len(set(p.text.lower() for p in pronouns))
        }
    
    def analyze_discourse_markers(self, doc):
        """Analyze usage of discourse markers"""
        text_lower = doc.text.lower()
        marker_counts = defaultdict(int)
        
        # Count occurrences of each type of discourse marker
        for marker_type, markers in self.discourse_markers.items():
            for marker in markers:
                count = len(re.findall(r'\b' + re.escape(marker) + r'\b', text_lower))
                marker_counts[marker_type] += count
                
        total_markers = sum(marker_counts.values())
        word_count = len(doc.text.split())
        
        return {
            'marker_density': total_markers / word_count,
            #'marker_distribution': dict(marker_counts),
            'marker_diversity': len([t for t, c in marker_counts.items() if c > 0])
        }
    
    def calculate_logical_connections(self, doc):
        """Calculate strength of logical connections between ideas"""
        # Split into sentences
        sentences = [sent.text.strip() for sent in doc.sents]
        print(sentences)
        # Get BERT embeddings for sentences
        embeddings = []
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(sentence, return_tensors="pt", 
                                      padding=True, truncation=True, max_length=512)
                outputs = self.model(**inputs)
                embeddings.append(outputs.last_hidden_state[:, 0, :].numpy())
        
        embeddings = np.vstack(embeddings)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create a graph from similarities
        G = nx.Graph()
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                if similarity_matrix[i, j] > 0.5:  # Threshold for connection
                    G.add_edge(i, j, weight=similarity_matrix[i, j])
        
        return {
            'avg_connection_strength': np.mean(similarity_matrix),
            'connection_density': nx.density(G),
        }
    
    def extract_all_features(self, doc):
        """Extract all contextual relevance features"""
        # Get all feature sets
        entity_features = self.extract_entity_relationships(doc)
        anaphora_features = self.analyze_anaphora(doc)
        discourse_features = self.analyze_discourse_markers(doc)
        #logical_features = self.calculate_logical_connections(doc)
        
        # Combine all features
        all_features = [
            entity_features, #entity_features
            anaphora_features, #anaphora_features
            discourse_features, #discourse_features
            #logical_features #logical_features
        ]
        
        return all_features
    
class NaturalTextPatternAnalyzer:
    def __init__(self):
        self.common_transitions = ['however', 'but', 'therefore', 'thus', 'moreover', 'furthermore']
        
    def analyze_sentence_variation(self, doc):
        """
        Analyzes natural variation in sentence structure and length
        """
        sentences = list(doc.sents)
        
        # Length variations
        sent_lengths = [len(sent) for sent in sentences]
        
        # Structure variations - analyze different sentence beginnings
        beginnings = defaultdict(int)
        for sent in sentences:
            first_token = sent[0]
            beginnings[first_token.pos_] += 1
            
        # Clause complexity
        clause_counts = []
        for sent in sentences:
            clauses = len([token for token in sent 
                         if token.dep_ in ['ccomp', 'xcomp', 'advcl']])
            clause_counts.append(clauses)
            
        return {
            'length_variance': np.var(sent_lengths),
            'length_cv': variation(sent_lengths),  # coefficient of variation
            'unique_beginnings_ratio': len(beginnings) / len(sentences),
            'avg_clause_complexity': np.mean(clause_counts),
            'clause_complexity_variance': np.var(clause_counts)
        }
    
    def analyze_idea_flow(self, doc):
        """
        Analyzes organic flow between ideas
        """
        sentences = list(doc.sents)
        
        # Analyze transition words
        transition_counts = defaultdict(int)
        for sent in sentences:
            sent_text = sent.text.lower()
            for trans in self.common_transitions:
                if trans in sent_text:
                    transition_counts[trans] += 1
        
        # Analyze topic continuity
        topic_shifts = 0
        prev_nouns = set()
        for sent in sentences:
            current_nouns = set(token.text for token in sent if token.pos_ == 'NOUN')
            if prev_nouns and not (prev_nouns & current_nouns):  # No common nouns
                topic_shifts += 1
            prev_nouns = current_nouns
            
        return {
            'transition_density': sum(transition_counts.values()) / len(sentences),
            'unique_transitions': len(transition_counts),
            'topic_shift_ratio': topic_shifts / (len(sentences) - 1) if len(sentences) > 1 else 0
        }
    
    def analyze_information_density(self, doc):
        """
        Analyzes how naturally information is presented
        """
        sentences = list(doc.sents)
        
        # Calculate information density metrics
        content_words_per_sent = []
        for sent in sentences:
            content_words = len([token for token in sent 
                               if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']])
            content_words_per_sent.append(content_words)
            
        # Use TF-IDF to measure information distribution
        vectorizer = TfidfVectorizer()
        sentence_texts = [sent.text for sent in sentences]
        tfidf_matrix = vectorizer.fit_transform(sentence_texts)
        
        return {
            'avg_content_density': np.mean(content_words_per_sent),
            'tfidf_variance': np.var(tfidf_matrix.toarray().sum(axis=1)),
        }
    
    def analyze_writing_imperfections(self, doc):
        """
        Analyzes natural imperfections in writing style
        """
        # Analyze sentence fragments
        fragments = len([sent for sent in doc.sents 
                        if not any(token.dep_ == 'ROOT' and token.pos_ == 'VERB' 
                                 for token in sent)])
        
        # Analyze repeated words (excluding stopwords)
        words = [token.text.lower() for token in doc if not token.is_stop]
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
            
        # Analyze parenthetical expressions and dashes
        parenthetical_count = len(re.findall(r'\(.*?\)', doc.text))
        dash_count = len(re.findall(r'—|--', doc.text))
        
        return {
            'fragment_ratio': fragments / len(list(doc.sents)),
            'word_repetition_rate': len([w for w, c in word_freq.items() if c > 1]) / len(words),
            'parenthetical_density': parenthetical_count / len(list(doc.sents)),
            'dash_usage_rate': dash_count / len(list(doc.sents))
        }
    
    def analyze_presentation_formality(self, doc):
        """
        Analyzes rigidity/formulaic nature of presentation
        """
        blob = TextBlob(doc.text)
        
        # Analyze sentence patterns
        sentence_patterns = []
        for sent in doc.sents:
            pattern = ''.join([token.pos_[0] for token in sent])
            sentence_patterns.append(pattern)
            
        # Calculate pattern repetition
        pattern_freq = defaultdict(int)
        for pattern in sentence_patterns:
            pattern_freq[pattern] += 1
            
        return {
            'subjectivity': blob.sentiment.subjectivity,
            'pattern_diversity': len(pattern_freq) / len(sentence_patterns),
            'max_pattern_repetition': max(pattern_freq.values()) / len(sentence_patterns),
            'avg_formality_score': self._calculate_formality_score(doc)
        }
        
    def _calculate_formality_score(self, doc) -> float:
        """
        Helper method to calculate text formality
        """
        formal_indicators = len([token for token in doc 
                               if token.pos_ in ['NOUN', 'ADJ'] 
                               or token.dep_ in ['compound', 'amod']])
        informal_indicators = len([token for token in doc 
                                 if token.pos_ in ['INTJ', 'PART'] 
                                 or token.dep_ == 'discourse'])
        return formal_indicators / (informal_indicators + 1)  # Avoid division by zero
    
    def analyze_all_patterns(self, doc):
        """
        Analyzes all natural text patterns
        """
        return [
            self.analyze_sentence_variation(doc), #sentence_variation
            #self.analyze_idea_flow(doc), #idea_flow
            self.analyze_information_density(doc), #information_density
            #self.analyze_writing_imperfections(doc), #writing_imperfections
            #self.analyze_presentation_formality(doc) #presentation_formality
        ]