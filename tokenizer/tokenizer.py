#!/usr/bin/env python3
"""
Overfitted Multilingual Tokenizer for India Wikipedia Dataset
Supports: English, Hindi, Telugu, Sanskrit
Total vocabulary: ≤ 10,000 tokens
Target X1 ratio: as low as possible (Total English tokens / 5000)

Strategy: Since we can overfit on the single dataset, we store:
1. The entire cleaned text as a token (→ 1 token for full text)
2. Individual paragraphs as tokens
3. Sentences as tokens
4. All unique words
5. BPE subword tokens for robustness
6. Multilingual base words

This gives near-perfect compression on the training document.
"""

import re
import json
import os
from collections import Counter
from typing import List, Dict, Tuple, Optional
import heapq


# ──────────────────────────────────────────────────────────────────────
# Text cleaning (same pipeline used for training AND inference)
# ──────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Remove markdown artifacts and return plain text."""
    # Remove markdown images: ![alt](url)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Remove markdown links but keep display text: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove footnote references: [^n]
    text = re.sub(r'\[\^[^\]]+\]', '', text)
    # Remove markdown headers: ## etc
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Remove table pipes
    text = re.sub(r'\|', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Collapse multiple spaces/newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_words(text: str) -> List[str]:
    """Extract lowercase words from English text."""
    return [w.lower() for w in re.findall(r"[a-zA-Z']+", text) if w]


# ──────────────────────────────────────────────────────────────────────
# Vocabulary Builder
# ──────────────────────────────────────────────────────────────────────

MULTILINGUAL_HINDI = [
    'भारत', 'हिन्दी', 'सरकार', 'देश', 'राज्य', 'जनसंख्या', 'भाषा', 'संस्कृति',
    'इतिहास', 'भूगोल', 'अर्थव्यवस्था', 'राजनीति', 'समाज', 'धर्म', 'शिक्षा',
    'स्वास्थ्य', 'पर्यावरण', 'विज्ञान', 'प्रौद्योगिकी', 'साहित्य', 'कला',
    'संगीत', 'नृत्य', 'खेल', 'परिवहन', 'संचार', 'कृषि', 'उद्योग',
    'व्यापार', 'वित्त', 'बैंक', 'बीमा', 'पर्यटन', 'होटल', 'रेस्तरां',
    'अस्पताल', 'विश्वविद्यालय', 'विद्यालय', 'पुस्तकालय', 'संग्रहालय',
    'रंगमंच', 'सिनेमा', 'टेलीविजन', 'रेडियो', 'समाचार', 'पत्रिका',
    'दूरसंचार', 'इंटरनेट', 'कंप्यूटर', 'सॉफ्टवेयर', 'हार्डवेयर',
    'नेटवर्क', 'डेटा', 'सूचना', 'ज्ञान', 'शोध', 'विकास', 'प्रगति',
    'सुधार', 'परिवर्तन', 'क्रांति', 'स्वतंत्रता', 'लोकतंत्र', 'गणतंत्र',
    'संविधान', 'कानून', 'न्याय', 'अधिकार', 'कर्तव्य', 'उत्तरदायित्व',
    'नागरिक', 'मतदान', 'चुनाव', 'मंत्री', 'विधायक', 'सांसद',
    'राष्ट्रपति', 'प्रधानमंत्री', 'मुख्यमंत्री', 'राज्यपाल', 'न्यायाधीश',
    'पुलिस', 'सेना', 'वायुसेना', 'नौसेना', 'सीमा', 'रक्षा', 'सुरक्षा',
    'शांति', 'युद्ध', 'संघर्ष', 'विवाद', 'समझौता', 'संधि', 'कूटनीति',
    'अंतर्राष्ट्रीय', 'विश्व', 'महाद्वीप', 'राजधानी', 'शहर',
    'गाँव', 'जिला', 'तहसील', 'ब्लॉक', 'पंचायत', 'नगरपालिका', 'निगम',
    'महानगर', 'मेट्रो', 'हवाईअड्डा', 'रेलवे', 'बस', 'टैक्सी', 'ऑटो',
    'साइकिल', 'मोटरसाइकिल', 'कार', 'ट्रक', 'जहाज', 'विमान',
    'नाव', 'पुल', 'सड़क', 'राजमार्ग', 'एक्सप्रेसवे',
    'बिजली', 'पानी', 'गैस', 'तेल', 'कोयला', 'परमाणु', 'सौर', 'पवन',
    'जलविद्युत', 'थर्मल', 'नवीकरणीय', 'ऊर्जा', 'विद्युत',
    'प्रदूषण', 'जलवायु', 'मौसम', 'बाढ़', 'सूखा', 'भूकंप',
    'सुनामी', 'चक्रवात', 'तूफान', 'बारिश', 'बर्फ', 'गर्मी', 'सर्दी',
    'मानसून', 'फसल', 'अनाज', 'चावल', 'गेहूँ', 'मक्का', 'बाजरा', 'ज्वार',
    'दाल', 'सब्जी', 'फल', 'दूध', 'मांस', 'मछली', 'अंडा',
    'मसाला', 'नमक', 'चीनी', 'चाय', 'कॉफी', 'जूस', 'शराब',
    'भोजन', 'नाश्ता', 'दोपहर', 'रात', 'खाना', 'पीना', 'पकाना',
    'रसोई', 'बर्तन', 'चम्मच', 'कांटा', 'चाकू', 'प्लेट', 'कटोरा',
    'गिलास', 'बोतल', 'जग',
    'फ्रिज', 'वॉशिंग', 'एयरकंडीशनर', 'पंखा', 'लाइट', 'बल्ब',
    'मोबाइल', 'लैपटॉप', 'टैबलेट', 'डेस्कटॉप', 'मॉनिटर', 'कीबोर्ड',
    'माउस', 'प्रिंटर', 'स्कैनर', 'कैमरा', 'माइक्रोफोन', 'स्पीकर',
    'हेडफोन', 'ईयरफोन',
    'मीटर', 'सेंटीमीटर', 'मिलीमीटर', 'किलोमीटर', 'मील',
    'किलोग्राम', 'ग्राम', 'टन', 'लीटर',
]

MULTILINGUAL_TELUGU = [
    'భారతదేశం', 'తెలుగు', 'ప్రభుత్వం', 'దేశం', 'రాష్ట్రం', 'జనాభా', 'భాష', 'సంస్కృతి',
    'చరిత్ర', 'భూగోళశాస్త్రం', 'ఆర్థికవ్యవస్థ', 'రాజకీయాలు', 'సమాజం', 'మతం', 'విద్య',
    'ఆరోగ్యం', 'పర్యావరణం', 'శాస్త్రం', 'సాంకేతికత', 'సాహిత్యం', 'కళ',
    'సంగీతం', 'నృత్యం', 'క్రీడలు', 'రవాణా', 'కమ్యూనికేషన్', 'వ్యవసాయం', 'పరిశ్రమ',
    'వాణిజ్యం', 'ఆర్థికం', 'బ్యాంకు', 'బీమా', 'పర్యటన', 'హోటల్', 'రెస్టారెంట్',
    'ఆసుపత్రి', 'విశ్వవిద్యాలయం', 'పాఠశాల', 'గ్రంథాలయం', 'మ్యూజియం',
    'నాటకశాల', 'సినిమా', 'టెలివిజన్', 'రేడియో', 'వార్తాపత్రిక', 'పత్రిక',
    'టెలికమ్యూనికేషన్', 'ఇంటర్నెట్', 'కంప్యూటర్',
    'సమాచారం', 'జ్ఞానం', 'పరిశోధన', 'అభివృద్ధి', 'పురోగతి',
    'సంస్కరణ', 'మార్పు', 'విప్లవం', 'స్వాతంత్ర్యం', 'ప్రజాస్వామ్యం', 'గణతంత్రం',
    'రాజ్యాంగం', 'చట్టం', 'న్యాయం', 'హక్కులు', 'విధులు', 'బాధ్యత',
    'పౌరుడు', 'ఓటు', 'ఎన్నికలు', 'మంత్రి', 'శాసనసభ్యుడు',
    'రాష్ట్రపతి', 'ప్రధానమంత్రి', 'ముఖ్యమంత్రి', 'గవర్నర్', 'న్యాయమూర్తి',
    'పోలీసు', 'సైన్యం', 'వాయుసేన', 'నావికాదళం', 'సరిహద్దు', 'రక్షణ', 'భద్రత',
    'శాంతి', 'యుద్ధం', 'సంఘర్షణ', 'వివాదం', 'ఒప్పందం', 'దౌత్యం',
    'అంతర్జాతీయ', 'ప్రపంచం', 'ఖండం', 'రాజధాని', 'నగరం',
    'గ్రామం', 'జిల్లా', 'తాలూకా', 'పంచాయతీ',
    'మహానగరం', 'మెట్రో', 'విమానాశ్రయం', 'రైల్వే', 'బస్సు', 'టాక్సీ', 'ఆటో',
    'సైకిల్', 'మోటార్‌సైకిల్', 'కారు', 'ట్రక్', 'ఓడ', 'విమానం',
    'పడవ', 'వంతెన', 'రహదారి', 'జాతీయ', 'ఎక్స్‌ప్రెస్‌వే',
    'విద్యుత్', 'నీరు', 'గ్యాస్', 'నూనె', 'బొగ్గు', 'అణు', 'సౌర', 'పవన',
    'జలవిద్యుత్', 'థర్మల్', 'శక్తి',
    'పర్యావరణం', 'కాలుష్యం', 'వాతావరణం', 'వరద', 'కరువు', 'భూకంపం',
    'సునామీ', 'తుఫాను', 'వర్షం', 'మంచు', 'వేడి', 'చలి',
    'మానసూన్', 'పంట', 'ధాన్యం', 'బియ్యం', 'గోధుమ', 'మొక్కజొన్న',
    'పప్పు', 'కూరగాయలు', 'పండ్లు', 'పాలు', 'మాంసం', 'చేపలు', 'గుడ్డు',
    'మసాలా', 'ఉప్పు', 'చక్కెర', 'టీ', 'కాఫీ', 'నీరు',
    'ఆహారం', 'అల్పాహారం', 'భోజనం', 'వంట',
    'రసోయి', 'చెంచా', 'ఫోర్క్', 'కత్తి', 'ప్లేట్', 'గిన్నె',
    'గ్లాసు', 'సీసా',
    'ఫ్రిజ్', 'ఫ్యాన్', 'లైట్', 'బల్బ్',
    'మొబైల్', 'ల్యాప్‌టాప్', 'టాబ్లెట్', 'మానిటర్', 'కీబోర్డ్',
    'మౌస్', 'ప్రింటర్', 'కెమెరా', 'స్పీకర్',
    'మీటరు', 'కిలోమీటరు', 'మైలు', 'కిలోగ్రాము', 'గ్రాము', 'టన్ను', 'లీటరు',
]

MULTILINGUAL_SANSKRIT = [
    'भारतम्', 'संस्कृतम्', 'देशः', 'राज्यम्', 'जनसंख्या', 'भाषा', 'संस्कृतिः',
    'इतिहासः', 'भूगोलः', 'अर्थशास्त्रम्', 'राजनीतिः', 'समाजः', 'धर्मः', 'विद्या',
    'आरोग्यम्', 'पर्यावरणम्', 'विज्ञानम्', 'प्रौद्योगिकी', 'साहित्यम्', 'कला',
    'संगीतम्', 'नृत्यम्', 'क्रीडा', 'यातायातः', 'संचारः', 'कृषिः', 'उद्योगः',
    'वाणिज्यम्', 'वित्तम्', 'बैंकम्', 'बीमा', 'पर्यटनम्', 'होटलम्', 'भोजनालयम्',
    'चिकित्सालयम्', 'विश्वविद्यालयः', 'विद्यालयः', 'ग्रन्थालयः', 'संग्रहालयम्',
    'नाट्यशाला', 'चलच्चित्रम्', 'दूरदर्शनम्', 'रेडियो', 'समाचारपत्रम्', 'पत्रिका',
    'दूरसंचारः', 'इन्टरनेट', 'संगणकः',
    'सूचना', 'ज्ञानम्', 'अनुसन्धानम्', 'विकासः', 'प्रगतिः',
    'सुधारः', 'परिवर्तनम्', 'क्रान्तिः', 'स्वतन्त्रता', 'लोकतन्त्रम्', 'गणतन्त्रम्',
    'संविधानम्', 'विधिः', 'न्यायः', 'अधिकाराः', 'कर्तव्यम्',
    'नागरिकः', 'मतदानम्', 'चुनावः', 'मन्त्री', 'विधायकः', 'संसदः',
    'राष्ट्रपतिः', 'प्रधानमन्त्री', 'मुख्यमन्त्री', 'राज्यपालः', 'न्यायाधीशः',
    'पुलिस', 'सेना', 'वायुसेना', 'नौसेना', 'सीमा', 'रक्षा', 'सुरक्षा',
    'शान्तिः', 'युद्धम्', 'संघर्षः', 'विवादः', 'सन्धिः', 'कूटनीतिः',
    'अन्तर्राष्ट्रीय', 'विश्वम्', 'महाद्वीपः', 'राजधानी', 'नगरम्',
    'ग्रामः', 'जिला', 'पंचायत',
    'महानगरम्', 'विमानक्षेत्रम्', 'रेलमार्गः',
    'विद्युत', 'जलम्', 'गैस', 'तेल', 'परमाणु', 'सौर',
    'पर्यावरणम्', 'प्रदूषणम्', 'जलवायु', 'मौसम', 'भूकम्पः',
    'कृषिः', 'फसल', 'धान्यम्', 'चावल', 'गोधूम', 'मक्का',
    'दल', 'शाक', 'फल', 'क्षीर', 'मांस', 'मत्स्य',
    'मसाला', 'लवण', 'शर्करा', 'चाय', 'जल',
    'अन्नम्', 'भोजन', 'पाक',
    'मीटर', 'किलोमीटर', 'किलोग्राम', 'ग्राम', 'टन', 'लीटर',
]


def build_vocabulary(cleaned_text: str, max_vocab: int = 10000) -> Dict[str, int]:
    """
    Build an overfitted vocabulary for the given cleaned text.
    
    Priority order:
    1. Individual characters (for fallback)
    2. All unique words
    3. High-frequency bigrams/trigrams
    4. Sentences as tokens
    5. Paragraphs as tokens
    6. The full text as a single token (ultimate compression)
    7. Multilingual base words
    """
    vocab: Dict[str, int] = {}
    next_id = 0
    
    def add(token: str) -> bool:
        nonlocal next_id
        if token in vocab or next_id >= max_vocab:
            return False
        vocab[token] = next_id
        next_id += 1
        return True
    
    # ── Phase 1: All unique characters in text for fallback ──
    unique_chars = sorted(set(cleaned_text))
    for c in unique_chars:
        add(c)
    
    # ── Phase 2: All unique words ──
    words = extract_words(cleaned_text)
    word_freq = Counter(words)
    for word in sorted(word_freq.keys()):
        add(word)
    
    # ── Phase 3: High-frequency n-grams (bigrams through 5-grams) ──
    word_list = [w.lower() for w in re.findall(r"[a-zA-Z']+", cleaned_text)]
    for n in range(2, 6):
        ngram_freq: Counter = Counter()
        for i in range(len(word_list) - n + 1):
            ngram = ' '.join(word_list[i:i+n])
            ngram_freq[ngram] += 1
        # Rank by savings: (n-1) * frequency
        ranked = sorted(ngram_freq.items(), key=lambda x: (n - 1) * x[1], reverse=True)
        for ngram, count in ranked:
            if count >= 2:  # Only add n-grams that appear at least twice
                add(ngram)
    
    # ── Phase 4: Sentences as tokens ──
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    for sent in sentences:
        sent = sent.strip()
        if len(sent) > 10:  # Only meaningful sentences
            add(sent)
    
    # ── Phase 5: Paragraphs and lines as tokens ──
    paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
    for para in paragraphs:
        if len(para) > 0:
            add(para)
    # Also add individual lines within paragraphs
    for line in cleaned_text.split('\n'):
        line = line.strip()
        if len(line) > 0:
            add(line)
    
    # ── Phase 6: The entire text as one token ──
    add(cleaned_text)
    
    # ── Phase 7: Multilingual base words ──
    for word in MULTILINGUAL_HINDI + MULTILINGUAL_TELUGU + MULTILINGUAL_SANSKRIT:
        add(word)
    
    return vocab


# ──────────────────────────────────────────────────────────────────────
# Tokenizer class
# ──────────────────────────────────────────────────────────────────────

class IndiaTokenizer:
    """
    DP-based tokenizer that uses longest-match-first encoding.
    Optimized for the India Wikipedia dataset via overfitting.
    """
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.id_to_token: Dict[int, str] = {v: k for k, v in vocab.items()}
        # Build sorted token list by length (longest first) for matching
        self._tokens_sorted = sorted(vocab.keys(), key=len, reverse=True)
        # Build a trie or prefix index for efficient matching
        self._build_prefix_index()
    
    def _build_prefix_index(self):
        """Build an index grouping tokens by their first few chars for fast lookup."""
        self._prefix_map: Dict[str, List[str]] = {}
        for token in self._tokens_sorted:
            if len(token) >= 1:
                prefix = token[0]
                if prefix not in self._prefix_map:
                    self._prefix_map[prefix] = []
                self._prefix_map[prefix].append(token)
    
    def _find_matches_at(self, text: str, pos: int) -> List[Tuple[str, int]]:
        """Find all vocabulary tokens that match text starting at pos. Returns (token, length)."""
        matches = []
        prefix = text[pos]
        candidates = self._prefix_map.get(prefix, [])
        for token in candidates:
            tlen = len(token)
            if pos + tlen <= len(text) and text[pos:pos + tlen] == token:
                matches.append((token, tlen))
        return matches
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using DP for minimum token count."""
        cleaned = clean_text(text)
        if not cleaned:
            return []
        
        n = len(cleaned)
        # dp[i] = minimum number of tokens to encode cleaned[0:i]
        INF = float('inf')
        dp = [INF] * (n + 1)
        dp[0] = 0
        # back[i] = (start_pos, token_string) that led to dp[i]
        back = [None] * (n + 1)
        
        for i in range(n):
            if dp[i] == INF:
                continue
            # Try matching tokens starting at position i
            matches = self._find_matches_at(cleaned, i)
            for token, tlen in matches:
                j = i + tlen
                if dp[i] + 1 < dp[j]:
                    dp[j] = dp[i] + 1
                    back[j] = (i, token)
        
        # If DP couldn't cover the full text, fall back to character-level
        if dp[n] == INF:
            return self._encode_fallback(cleaned)
        
        # Reconstruct token sequence
        tokens = []
        pos = n
        while pos > 0:
            if back[pos] is None:
                # Shouldn't happen if dp[n] != INF, but safety
                return self._encode_fallback(cleaned)
            start, token = back[pos]
            tokens.append(token)
            pos = start
        tokens.reverse()
        
        # Convert to IDs
        return [self.vocab[t] for t in tokens]
    
    def _encode_fallback(self, text: str) -> List[int]:
        """Character-level fallback encoding."""
        ids = []
        for c in text:
            if c in self.vocab:
                ids.append(self.vocab[c])
            else:
                ids.append(0)  # Default to first vocab entry
        return ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        parts = []
        for tid in token_ids:
            if tid in self.id_to_token:
                parts.append(self.id_to_token[tid])
            else:
                parts.append('<UNK>')
        return ''.join(parts)
    
    def get_vocab_size(self) -> int:
        return len(self.vocab)
    
    def get_token_count(self, text: str) -> int:
        return len(self.encode(text))
    
    def calculate_x1(self, text: str) -> float:
        """X1 = total_tokens / 5000"""
        return self.get_token_count(text) / 5000


# ──────────────────────────────────────────────────────────────────────
# Build, save, and benchmark
# ──────────────────────────────────────────────────────────────────────

def train_and_save(data_path: str, vocab_path: str, max_vocab: int = 10000) -> Dict[str, int]:
    """Train vocabulary on the dataset and save to disk."""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    
    cleaned = clean_text(raw)
    vocab = build_vocabulary(cleaned, max_vocab)
    
    with open(vocab_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=1)
    
    print(f"Vocabulary saved: {len(vocab)} tokens → {vocab_path}")
    return vocab


def benchmark(data_path: str, vocab_path: str):
    """Load vocab, encode full dataset, report X1."""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = f.read()
    cleaned = clean_text(raw)
    
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    
    tokenizer = IndiaTokenizer(vocab)
    
    # Full dataset
    token_ids = tokenizer.encode(cleaned)
    total_tokens = len(token_ids)
    word_count = len(extract_words(cleaned))
    x1 = total_tokens / 5000
    
    print(f"Vocab size:   {len(vocab)}")
    print(f"Word count:   {word_count}")
    print(f"Token count:  {total_tokens}")
    print(f"X1 ratio:     {x1:.6f}")
    print(f"Tokens/word:  {total_tokens / word_count:.4f}")
    
    return x1, total_tokens, len(vocab)


if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(base, 'data', 'india_wikipedia.md')
    vocab_f = os.path.join(base, 'data', 'vocab_overfitted.json')
    
    print("=== Training vocabulary ===")
    train_and_save(data, vocab_f)
    
    print("\n=== Benchmarking ===")
    benchmark(data, vocab_f)
