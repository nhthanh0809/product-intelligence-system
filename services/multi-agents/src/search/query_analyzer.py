"""
Query analyzer for determining optimal search strategy.

Analyzes queries to detect:
- Query type (brand+model, model number, generic, etc.)
- Model numbers
- Brands
- Sections (reviews, specs, features)
- Optimal keyword/semantic weights
"""

import re
from dataclasses import dataclass, field
from enum import Enum


class QueryType(str, Enum):
    """Detected query type for weight adjustment."""
    BRAND_MODEL = "brand_model"      # "Sony WH-1000XM5"
    MODEL_NUMBER = "model_number"    # "WH-1000XM5"
    SHORT_TITLE = "short_title"      # "wireless headphones"
    GENERIC = "generic"              # "good headphones for travel"
    SECTION = "section"              # "reviews of Sony headphones"


@dataclass
class QueryAnalysis:
    """Result of analyzing a query."""
    query_type: QueryType
    clean_query: str
    model_numbers: list[str] = field(default_factory=list)
    has_brand: bool = False
    detected_brand: str = ""
    detected_section: str | None = None
    keyword_weight: float = 0.65
    semantic_weight: float = 0.35

    @property
    def model_boost(self) -> float:
        """Boost factor for model number detection."""
        return 1.6 if self.model_numbers else 1.0

    @property
    def brand_boost(self) -> float:
        """Boost factor for brand detection."""
        return 1.3 if self.has_brand else 1.0


class QueryAnalyzer:
    """Analyze queries to determine optimal search strategy.

    Based on search-flow experiments (Jan 2026):
    - Brand+model queries: keyword weight 0.75
    - Model number queries: keyword weight 0.70
    - Short title queries: keyword weight 0.65
    - Generic queries: semantic weight 0.60
    """

    # Known brands for detection
    KNOWN_BRANDS = [
        "Sony", "Bose", "Apple", "Samsung", "LG", "Anker", "JBL", "Sennheiser",
        "Dell", "HP", "Lenovo", "Microsoft", "Google", "Amazon", "Kindle",
        "DeWalt", "Milwaukee", "Makita", "Bosch", "Ryobi", "Black+Decker",
        "Canon", "Nikon", "Fujifilm", "Dyson", "Shark", "Roomba", "iRobot",
        "KitchenAid", "Ninja", "Instant Pot", "Vitamix", "Philips", "Panasonic",
        "Beats", "Logitech", "Razer", "Corsair", "ASUS", "Acer", "MSI",
        "Nintendo", "PlayStation", "Xbox", "Fitbit", "Garmin", "GoPro",
    ]

    # Section keywords for targeted searches
    SECTION_KEYWORDS = {
        "reviews": ["review", "reviews", "rating", "ratings", "what do people say", "opinions"],
        "specs": ["specs", "specifications", "technical", "dimensions", "weight", "size"],
        "features": ["features", "capabilities", "what can it do", "functions"],
        "description": ["description", "about", "overview", "what is"],
        "use_cases": ["use case", "best for", "good for", "suited for", "who should buy"],
    }

    # Model number patterns
    MODEL_PATTERNS = [
        r'\b[A-Z]{2,4}[-]?\d{3,4}[A-Z]?\d*\b',  # WH-1000XM5, RTX4090
        r'\b[A-Z]\d{2,4}[A-Z]?\b',               # A15, M2
        r'\b\d{3,4}[A-Z]{1,2}\b',                # 990Pro
        r'\b[A-Z]{1,2}\d{1,2}\s*(?:Pro|Max|Plus|Ultra|Mini|Air)\b',  # M2 Pro, A16 Max
    ]

    @classmethod
    def analyze(cls, query: str) -> QueryAnalysis:
        """Analyze query to determine type and optimal weights.

        Args:
            query: Raw search query

        Returns:
            QueryAnalysis with type, weights, and extracted entities
        """
        clean_query = cls._clean_query(query)
        model_numbers = cls._extract_model_numbers(query)
        has_brand, detected_brand = cls._detect_brand(query)
        detected_section = cls._detect_section(query)
        words = clean_query.split()

        # Determine query type and weights based on experiments
        if detected_section:
            query_type = QueryType.SECTION
            kw_weight, sem_weight = 0.40, 0.60
        elif model_numbers:
            query_type = QueryType.MODEL_NUMBER
            kw_weight, sem_weight = 0.75, 0.25
        elif has_brand and len(words) <= 5:
            query_type = QueryType.BRAND_MODEL
            kw_weight, sem_weight = 0.70, 0.30
        elif len(words) <= 3:
            query_type = QueryType.SHORT_TITLE
            kw_weight, sem_weight = 0.65, 0.35
        elif len(words) <= 6:
            query_type = QueryType.SHORT_TITLE
            kw_weight, sem_weight = 0.60, 0.40
        else:
            query_type = QueryType.GENERIC
            kw_weight, sem_weight = 0.40, 0.60

        return QueryAnalysis(
            query_type=query_type,
            clean_query=clean_query,
            model_numbers=model_numbers,
            has_brand=has_brand,
            detected_brand=detected_brand,
            detected_section=detected_section,
            keyword_weight=kw_weight,
            semantic_weight=sem_weight,
        )

    @classmethod
    def _clean_query(cls, query: str) -> str:
        """Clean common artifacts from queries."""
        # Remove markdown artifacts
        query = re.sub(r'\*+', ' ', query)
        query = re.sub(r'\*\*([^*]+)\*\*', r'\1', query)
        # Remove common question prefixes
        prefixes = [
            r'^(find|search for|looking for|show me|get me|i want|i need)\s+',
            r'^(what is|what are|how to|where can i find)\s+',
        ]
        for prefix in prefixes:
            query = re.sub(prefix, '', query, flags=re.IGNORECASE)
        # Normalize whitespace
        query = ' '.join(query.split())
        return query.strip()

    @classmethod
    def _extract_model_numbers(cls, query: str) -> list[str]:
        """Extract model numbers from query."""
        model_numbers = []
        for pattern in cls.MODEL_PATTERNS:
            matches = re.findall(pattern, query, re.IGNORECASE)
            model_numbers.extend(matches)
        return list(set(model_numbers))

    @classmethod
    def _detect_brand(cls, query: str) -> tuple[bool, str]:
        """Detect if query contains a known brand."""
        query_lower = query.lower()
        for brand in cls.KNOWN_BRANDS:
            if brand.lower() in query_lower:
                return True, brand
        return False, ""

    @classmethod
    def _detect_section(cls, query: str) -> str | None:
        """Detect if query is asking about a specific section."""
        query_lower = query.lower()
        for section, keywords in cls.SECTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return section
        return None

    @classmethod
    def get_weight_recommendation(cls, query_type: QueryType) -> tuple[float, float]:
        """Get recommended keyword/semantic weights for a query type.

        Returns:
            Tuple of (keyword_weight, semantic_weight)
        """
        weights = {
            QueryType.BRAND_MODEL: (0.75, 0.25),
            QueryType.MODEL_NUMBER: (0.70, 0.30),
            QueryType.SHORT_TITLE: (0.65, 0.35),
            QueryType.GENERIC: (0.40, 0.60),
            QueryType.SECTION: (0.40, 0.60),
        }
        return weights.get(query_type, (0.65, 0.35))
