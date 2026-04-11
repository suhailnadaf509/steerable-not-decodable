"""
Task Battery with Principled Taxonomy (Redesigned per EXPERIMENT_REDESIGN_SPEC.md)
==================================================================================
12 tasks spanning 5 categories, designed for a range of expected IID accuracies.
Each task has:
  - Unambiguous ground truth
  - 60-95 input-output pairs
  - Clear accuracy criterion
  - Documented expected difficulty
  - 8 task-specific templates (T1-T8) covering natural / symbolic / question / formal styles

Taxonomy (informed by Todd et al. 2024, Hendel et al. 2023):
  1. Lexical Retrieval      -- answer is a stored word-level association
  2. Factual Retrieval      -- answer is a specific world-knowledge fact
  3. Morphological Transform -- answer requires grammatical rule application
  4. Character / Surface     -- answer depends on character-level manipulation
  5. Compositional / Semantic -- multi-token, semantic-level rewrite

Tasks removed from prior version:
  - present_to_gerund (redundant with past_tense)
  - singular_past (literally past_tense with different templates)

Tasks added:
  - reverse_word (Character/Surface — three-point gradient with capitalize + first_letter)
  - object_color (Factual Retrieval — different relational structure from country_capital)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TaskCategory(Enum):
    LEXICAL_RETRIEVAL = "lexical_retrieval"
    FACTUAL_RETRIEVAL = "factual_retrieval"
    MORPHOLOGICAL_TRANSFORM = "morphological_transform"
    CHARACTER_SURFACE = "character_surface"
    COMPOSITIONAL_SEMANTIC = "compositional_semantic"


class TemplateStyle(Enum):
    """Style category for templates. T1-T2 = Natural, T3-T4 = Symbolic, etc."""
    NATURAL = "natural"
    SYMBOLIC = "symbolic"
    QUESTION = "question"
    FORMAL = "formal"


# Template ID -> Style mapping (convention from spec Section 3.3)
TEMPLATE_STYLE_MAP: Dict[str, TemplateStyle] = {
    "T1": TemplateStyle.NATURAL,
    "T2": TemplateStyle.NATURAL,
    "T3": TemplateStyle.SYMBOLIC,
    "T4": TemplateStyle.SYMBOLIC,
    "T5": TemplateStyle.QUESTION,
    "T6": TemplateStyle.QUESTION,
    "T7": TemplateStyle.FORMAL,
    "T8": TemplateStyle.FORMAL,
}


class AccuracyMode(Enum):
    """How to evaluate model output against ground truth."""
    EXACT = "exact"               # lowercased exact match
    SUBSTRING = "substring"       # ground truth appears in output (case-insensitive)
    CASE_SENSITIVE_SUB = "case_sensitive_substring"   # case-sensitive substring


@dataclass
class TaskSpec:
    """Complete specification for a single task."""
    name: str
    category: TaskCategory
    description: str
    expected_difficulty: str  # "easy", "medium", "hard"

    # Templates: dict of template_id -> template string with {X} placeholder
    templates: Dict[str, str]

    # Data pairs: list of (input, expected_output)
    pairs: List[Tuple[str, str]]

    # Evaluation
    accuracy_mode: AccuracyMode = AccuracyMode.SUBSTRING
    max_new_tokens: int = 5   # task-adaptive

    # Optional: list of alternative valid outputs for ambiguous tasks
    # Maps input -> list of valid outputs (including the primary one)
    alternative_outputs: Optional[Dict[str, List[str]]] = None

    def check_accuracy(self, input_val: str, model_output: str) -> bool:
        """Deterministic accuracy check for this task."""
        expected = self.get_ground_truth(input_val)
        if expected is None:
            return False

        output_clean = model_output.strip()
        expected_clean = expected.strip()

        if self.accuracy_mode == AccuracyMode.EXACT:
            return output_clean.lower() == expected_clean.lower()
        elif self.accuracy_mode == AccuracyMode.SUBSTRING:
            # Check primary output
            if expected_clean.lower() in output_clean.lower():
                return True
            # Check alternatives if available
            if self.alternative_outputs and input_val in self.alternative_outputs:
                for alt in self.alternative_outputs[input_val]:
                    if alt.strip().lower() in output_clean.lower():
                        return True
            return False
        elif self.accuracy_mode == AccuracyMode.CASE_SENSITIVE_SUB:
            return expected_clean in output_clean
        return False

    def get_ground_truth(self, input_val: str) -> Optional[str]:
        """Look up expected output for an input."""
        for inp, out in self.pairs:
            if inp == input_val:
                return out
        return None

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    @property
    def template_ids(self) -> List[str]:
        return sorted(self.templates.keys())

    @property
    def n_templates(self) -> int:
        return len(self.templates)

    def get_template_style(self, template_id: str) -> Optional[TemplateStyle]:
        """Get the style category for a template ID."""
        return TEMPLATE_STYLE_MAP.get(template_id)

    def validate(self) -> List[str]:
        """Return warnings about this task specification."""
        warnings = []
        if self.n_pairs < 60:
            warnings.append(
                f"Task '{self.name}' has only {self.n_pairs} pairs "
                f"(need >= 60 for 15 ICL + 50 test)"
            )
        if self.n_templates < 8:
            warnings.append(
                f"Task '{self.name}' has only {self.n_templates} templates (need >= 8)"
            )
        # Check for duplicate inputs
        inputs = [inp for inp, _ in self.pairs]
        if len(inputs) != len(set(inputs)):
            warnings.append(f"Task '{self.name}' has duplicate inputs")
        # Check all templates contain {X}
        for tid, tstr in self.templates.items():
            if "{X}" not in tstr:
                warnings.append(f"Task '{self.name}' template {tid} missing {{X}} placeholder")
        return warnings


# ===========================================================================
# Word lists and data for each task
# ===========================================================================

ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("hot", "cold"), ("big", "small"), ("fast", "slow"), ("happy", "sad"),
    ("light", "dark"), ("up", "down"), ("good", "bad"), ("old", "new"),
    ("hard", "soft"), ("high", "low"), ("rich", "poor"), ("strong", "weak"),
    ("long", "short"), ("wide", "narrow"), ("deep", "shallow"), ("loud", "quiet"),
    ("bright", "dim"), ("clean", "dirty"), ("dry", "wet"), ("empty", "full"),
    ("early", "late"), ("easy", "difficult"), ("far", "near"), ("thick", "thin"),
    ("heavy", "light"), ("rough", "smooth"), ("sharp", "dull"), ("sweet", "sour"),
    ("true", "false"), ("young", "old"), ("alive", "dead"), ("awake", "asleep"),
    ("beautiful", "ugly"), ("brave", "cowardly"), ("calm", "angry"),
    ("cheap", "expensive"), ("clever", "stupid"), ("cruel", "kind"),
    ("dangerous", "safe"), ("different", "same"), ("polite", "rude"),
    ("public", "private"), ("quick", "slow"), ("rare", "common"),
    ("right", "wrong"), ("simple", "complex"), ("straight", "curved"),
    ("tight", "loose"), ("ancient", "modern"), ("bitter", "sweet"),
    ("bold", "timid"), ("boring", "exciting"), ("broad", "narrow"),
    ("certain", "uncertain"), ("clear", "cloudy"), ("complete", "incomplete"),
    ("correct", "incorrect"), ("dense", "sparse"), ("fancy", "plain"),
    ("fierce", "gentle"), ("firm", "soft"), ("flat", "curved"),
    ("fresh", "stale"), ("friendly", "hostile"), ("generous", "selfish"),
    ("guilty", "innocent"), ("humble", "proud"), ("legal", "illegal"),
    ("major", "minor"), ("natural", "artificial"), ("normal", "abnormal"),
    ("open", "closed"), ("ordinary", "extraordinary"), ("passive", "active"),
    ("patient", "impatient"), ("peaceful", "violent"), ("permanent", "temporary"),
    ("positive", "negative"), ("powerful", "weak"), ("present", "absent"),
    ("proud", "humble"), ("pure", "impure"), ("real", "fake"),
    ("regular", "irregular"), ("rural", "urban"), ("serious", "funny"),
    ("silent", "noisy"), ("single", "married"), ("smooth", "rough"),
    ("solid", "liquid"), ("specific", "general"), ("steady", "unsteady"),
    ("strict", "lenient"), ("success", "failure"),
    ("visible", "invisible"),
]  # 95 pairs per spec

SYNONYM_PAIRS: List[Tuple[str, str]] = [
    ("happy", "glad"), ("sad", "unhappy"), ("big", "large"), ("small", "tiny"),
    ("fast", "quick"), ("slow", "sluggish"), ("smart", "clever"), ("dumb", "stupid"),
    ("pretty", "beautiful"), ("ugly", "hideous"), ("rich", "wealthy"),
    ("poor", "destitute"), ("brave", "courageous"), ("scared", "afraid"),
    ("angry", "furious"), ("calm", "serene"), ("old", "elderly"),
    ("young", "youthful"), ("strong", "powerful"), ("weak", "feeble"),
    ("kind", "benevolent"), ("cruel", "ruthless"), ("honest", "truthful"),
    ("evil", "wicked"), ("bright", "brilliant"), ("dark", "gloomy"),
    ("loud", "noisy"), ("quiet", "silent"), ("hard", "difficult"),
    ("easy", "simple"), ("clean", "spotless"), ("dirty", "filthy"),
    ("hot", "scorching"), ("cold", "freezing"), ("wet", "damp"),
    ("dry", "arid"), ("tall", "towering"), ("short", "brief"),
    ("wide", "broad"), ("thin", "slender"), ("deep", "profound"),
    ("empty", "vacant"), ("full", "complete"), ("new", "fresh"),
    ("ancient", "antique"), ("rough", "coarse"), ("smooth", "sleek"),
    ("sharp", "keen"), ("blunt", "dull"), ("sweet", "sugary"),
    ("sour", "tart"), ("strange", "odd"), ("normal", "ordinary"),
    ("rare", "uncommon"), ("common", "frequent"), ("polite", "courteous"),
    ("rude", "impolite"), ("generous", "charitable"), ("greedy", "avaricious"),
    ("humble", "modest"), ("proud", "arrogant"), ("patient", "tolerant"),
    ("lazy", "idle"), ("busy", "occupied"), ("free", "liberated"),
    ("serious", "solemn"), ("funny", "humorous"), ("famous", "renowned"),
    ("secret", "covert"), ("obvious", "apparent"), ("careful", "cautious"),
    ("careless", "negligent"), ("boring", "tedious"), ("interesting", "fascinating"),
    ("real", "genuine"), ("fake", "counterfeit"), ("fresh", "novel"),
    ("stale", "outdated"), ("tough", "resilient"), ("fragile", "delicate"),
    ("lively", "vibrant"), ("dead", "lifeless"), ("peaceful", "tranquil"),
    ("violent", "aggressive"), ("friendly", "amiable"), ("hostile", "antagonistic"),
    ("nervous", "anxious"), ("cheerful", "jovial"),
]  # 88 pairs per spec

# Alternative valid outputs for synonym (one-to-many mapping)
SYNONYM_ALTERNATIVES: Dict[str, List[str]] = {
    "happy": ["glad", "joyful", "cheerful", "content"],
    "sad": ["unhappy", "sorrowful", "melancholy", "gloomy"],
    "big": ["large", "huge", "enormous", "massive"],
    "small": ["tiny", "little", "miniature", "petite"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "smart": ["clever", "intelligent", "bright", "brilliant"],
    "pretty": ["beautiful", "attractive", "lovely", "gorgeous"],
    "brave": ["courageous", "bold", "fearless", "valiant"],
    "angry": ["furious", "mad", "irate", "enraged"],
    "strong": ["powerful", "mighty", "robust", "sturdy"],
}

HYPERNYM_PAIRS: List[Tuple[str, str]] = [
    ("dog", "animal"), ("cat", "animal"), ("rose", "flower"), ("oak", "tree"),
    ("car", "vehicle"), ("truck", "vehicle"), ("apple", "fruit"),
    ("banana", "fruit"), ("hammer", "tool"), ("saw", "tool"),
    ("guitar", "instrument"), ("piano", "instrument"), ("shirt", "clothing"),
    ("pants", "clothing"), ("chair", "furniture"), ("table", "furniture"),
    ("sparrow", "bird"), ("eagle", "bird"), ("salmon", "fish"),
    ("trout", "fish"), ("python", "snake"), ("cobra", "snake"),
    ("baseball", "sport"), ("soccer", "sport"), ("novel", "book"),
    ("poem", "literature"), ("gold", "metal"), ("silver", "metal"),
    ("diamond", "gem"), ("ruby", "gem"), ("wheat", "grain"),
    ("rice", "grain"), ("beer", "beverage"), ("wine", "beverage"),
    ("violin", "instrument"), ("drums", "instrument"), ("sword", "weapon"),
    ("spear", "weapon"), ("painting", "art"), ("sculpture", "art"),
    ("english", "language"), ("french", "language"), ("earth", "planet"),
    ("mars", "planet"), ("dollar", "currency"), ("euro", "currency"),
    ("oxygen", "element"), ("carbon", "element"), ("granite", "rock"),
    ("marble", "rock"), ("cotton", "fabric"), ("silk", "fabric"),
    ("waltz", "dance"), ("tango", "dance"), ("sonnet", "poem"),
    ("haiku", "poem"), ("jazz", "music"), ("opera", "music"),
    ("chess", "game"), ("poker", "game"), ("canoe", "boat"),
    ("yacht", "boat"), ("bicycle", "vehicle"), ("bus", "vehicle"),
    ("tiger", "animal"), ("lion", "animal"), ("daisy", "flower"),
    ("tulip", "flower"), ("maple", "tree"), ("pine", "tree"),
    ("orange", "fruit"), ("grape", "fruit"), ("drill", "tool"),
    ("wrench", "tool"), ("jacket", "clothing"), ("dress", "clothing"),
    ("sofa", "furniture"), ("desk", "furniture"), ("parrot", "bird"),
    ("penguin", "bird"), ("shark", "fish"), ("tuna", "fish"),
    ("tennis", "sport"), ("golf", "sport"), ("copper", "metal"),
    ("iron", "metal"),
]  # 86 pairs per spec

# Alternative valid outputs for hypernym
HYPERNYM_ALTERNATIVES: Dict[str, List[str]] = {
    "dog": ["animal", "mammal", "pet", "canine"],
    "cat": ["animal", "mammal", "pet", "feline"],
    "rose": ["flower", "plant"],
    "apple": ["fruit", "food"],
    "banana": ["fruit", "food"],
    "car": ["vehicle", "automobile", "transportation"],
    "guitar": ["instrument", "musical instrument"],
    "piano": ["instrument", "musical instrument"],
    "novel": ["book", "literature", "fiction"],
    "poem": ["literature", "writing", "poetry"],
}

COUNTRY_CAPITAL_PAIRS: List[Tuple[str, str]] = [
    ("France", "Paris"), ("Germany", "Berlin"), ("Japan", "Tokyo"),
    ("Italy", "Rome"), ("Spain", "Madrid"), ("Brazil", "Brasilia"),
    ("Canada", "Ottawa"), ("Australia", "Canberra"), ("China", "Beijing"),
    ("India", "Delhi"), ("Russia", "Moscow"), ("Mexico", "Mexico City"),
    ("Egypt", "Cairo"), ("Turkey", "Ankara"), ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"), ("Greece", "Athens"), ("Portugal", "Lisbon"),
    ("Austria", "Vienna"), ("Switzerland", "Bern"), ("Belgium", "Brussels"),
    ("Netherlands", "Amsterdam"), ("Ireland", "Dublin"),
    ("Argentina", "Buenos Aires"), ("Chile", "Santiago"),
    ("Colombia", "Bogota"), ("Peru", "Lima"), ("Thailand", "Bangkok"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Malaysia", "Kuala Lumpur"),
    ("Philippines", "Manila"), ("Nigeria", "Abuja"), ("Kenya", "Nairobi"),
    ("Morocco", "Rabat"), ("Cuba", "Havana"), ("Hungary", "Budapest"),
    ("Romania", "Bucharest"), ("Ukraine", "Kyiv"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Pakistan", "Islamabad"), ("Bangladesh", "Dhaka"),
    ("Nepal", "Kathmandu"), ("Myanmar", "Naypyidaw"), ("Cambodia", "Phnom Penh"),
    ("Jordan", "Amman"), ("Lebanon", "Beirut"), ("Israel", "Jerusalem"),
    ("Qatar", "Doha"), ("Oman", "Muscat"), ("Croatia", "Zagreb"),
    ("Serbia", "Belgrade"), ("Bulgaria", "Sofia"), ("Slovakia", "Bratislava"),
    ("Lithuania", "Vilnius"), ("Latvia", "Riga"), ("Estonia", "Tallinn"),
    ("Slovenia", "Ljubljana"), ("Iceland", "Reykjavik"),
    ("Luxembourg", "Luxembourg"), ("Malta", "Valletta"), ("Cyprus", "Nicosia"),
    ("Jamaica", "Kingston"), ("Trinidad", "Port of Spain"),
    ("Panama", "Panama City"), ("Uruguay", "Montevideo"),
    ("Paraguay", "Asuncion"), ("Bolivia", "Sucre"), ("Ecuador", "Quito"),
    ("Venezuela", "Caracas"), ("Senegal", "Dakar"), ("Ghana", "Accra"),
    ("Tanzania", "Dodoma"), ("Ethiopia", "Addis Ababa"), ("Uganda", "Kampala"),
    ("Mozambique", "Maputo"), ("Zimbabwe", "Harare"), ("Zambia", "Lusaka"),
    ("Botswana", "Gaborone"), ("Namibia", "Windhoek"),
    ("Madagascar", "Antananarivo"), ("Mongolia", "Ulaanbaatar"),
    ("Georgia", "Tbilisi"), ("Armenia", "Yerevan"), ("Azerbaijan", "Baku"),
    ("Albania", "Tirana"),
]  # 90 pairs per spec

PAST_TENSE_PAIRS: List[Tuple[str, str]] = [
    ("walk", "walked"), ("talk", "talked"), ("play", "played"),
    ("jump", "jumped"), ("look", "looked"), ("work", "worked"),
    ("call", "called"), ("help", "helped"), ("start", "started"),
    ("move", "moved"), ("live", "lived"), ("like", "liked"),
    ("want", "wanted"), ("need", "needed"), ("ask", "asked"),
    ("turn", "turned"), ("show", "showed"), ("learn", "learned"),
    ("change", "changed"), ("watch", "watched"), ("follow", "followed"),
    ("stop", "stopped"), ("create", "created"), ("open", "opened"),
    ("close", "closed"), ("laugh", "laughed"), ("pull", "pulled"),
    ("push", "pushed"), ("pick", "picked"), ("cook", "cooked"),
    ("clean", "cleaned"), ("wash", "washed"), ("fill", "filled"),
    ("mark", "marked"), ("fix", "fixed"), ("reach", "reached"),
    ("join", "joined"), ("serve", "served"), ("pass", "passed"),
    ("point", "pointed"), ("rest", "rested"), ("paint", "painted"),
    ("kick", "kicked"), ("miss", "missed"), ("rain", "rained"),
    ("climb", "climbed"), ("plant", "planted"), ("knock", "knocked"),
    ("drop", "dropped"), ("cross", "crossed"), ("add", "added"),
    ("count", "counted"), ("lift", "lifted"), ("save", "saved"),
    ("earn", "earned"), ("share", "shared"), ("dance", "danced"),
    ("smile", "smiled"), ("smoke", "smoked"), ("taste", "tasted"),
    ("touch", "touched"), ("pack", "packed"), ("print", "printed"),
    ("pour", "poured"), ("load", "loaded"), ("rush", "rushed"),
    ("crash", "crashed"), ("float", "floated"), ("fold", "folded"),
    ("guess", "guessed"), ("hunt", "hunted"), ("iron", "ironed"),
    ("joke", "joked"), ("land", "landed"), ("last", "lasted"),
    ("lock", "locked"), ("melt", "melted"), ("mix", "mixed"),
    ("note", "noted"), ("order", "ordered"), ("park", "parked"),
    ("place", "placed"), ("race", "raced"), ("sail", "sailed"),
    ("shop", "shopped"), ("sign", "signed"), ("test", "tested"),
    ("train", "trained"), ("trap", "trapped"), ("trust", "trusted"),
]  # 90 pairs per spec

PLURAL_PAIRS: List[Tuple[str, str]] = [
    ("cat", "cats"), ("dog", "dogs"), ("car", "cars"), ("book", "books"),
    ("tree", "trees"), ("house", "houses"), ("bird", "birds"), ("fish", "fish"),
    ("star", "stars"), ("rock", "rocks"), ("hill", "hills"), ("lake", "lakes"),
    ("road", "roads"), ("wall", "walls"), ("door", "doors"), ("hand", "hands"),
    ("foot", "feet"), ("eye", "eyes"), ("ear", "ears"), ("arm", "arms"),
    ("leg", "legs"), ("cup", "cups"), ("box", "boxes"), ("bag", "bags"),
    ("ball", "balls"), ("bell", "bells"), ("boat", "boats"), ("cake", "cakes"),
    ("card", "cards"), ("coin", "coins"), ("desk", "desks"), ("drum", "drums"),
    ("farm", "farms"), ("flag", "flags"), ("fork", "forks"), ("game", "games"),
    ("gift", "gifts"), ("girl", "girls"), ("goal", "goals"), ("hat", "hats"),
    ("horn", "horns"), ("joke", "jokes"), ("king", "kings"), ("lamp", "lamps"),
    ("leaf", "leaves"), ("lion", "lions"), ("mask", "masks"), ("meal", "meals"),
    ("moon", "moons"), ("nest", "nests"), ("note", "notes"), ("park", "parks"),
    ("path", "paths"), ("pipe", "pipes"), ("plan", "plans"), ("poem", "poems"),
    ("pool", "pools"), ("ring", "rings"), ("roof", "roofs"), ("rope", "ropes"),
    ("rule", "rules"), ("seed", "seeds"), ("ship", "ships"), ("shoe", "shoes"),
    ("sign", "signs"), ("song", "songs"), ("step", "steps"), ("tail", "tails"),
    ("tank", "tanks"), ("team", "teams"), ("tent", "tents"), ("test", "tests"),
    ("tool", "tools"), ("tour", "tours"), ("town", "towns"), ("trap", "traps"),
    ("trip", "trips"), ("tube", "tubes"), ("unit", "units"), ("vine", "vines"),
    ("wave", "waves"), ("wire", "wires"), ("wolf", "wolves"),
    ("word", "words"), ("zone", "zones"), ("hero", "heroes"),
    ("tomato", "tomatoes"), ("potato", "potatoes"), ("knife", "knives"),
    ("wife", "wives"),
]  # 90 pairs per spec

CAPITALIZE_WORDS: List[Tuple[str, str]] = [
    (w, w.upper()) for w in [
        "hello", "world", "python", "machine", "learning", "neural", "network",
        "data", "science", "algorithm", "computer", "program", "function",
        "variable", "class", "object", "method", "array", "string", "integer",
        "float", "boolean", "list", "dictionary", "tuple", "module", "package",
        "library", "framework", "model", "train", "test", "batch", "epoch",
        "loss", "gradient", "weight", "bias", "layer", "input", "output",
        "hidden", "activation", "attention", "transformer", "encoder", "decoder",
        "embedding", "token", "sequence", "vector", "matrix", "tensor", "scalar",
        "dimension", "shape", "size", "length", "width", "height", "depth",
        "kernel", "filter", "stride", "padding", "pooling", "dropout",
        "abstract", "concrete", "dynamic", "static", "private", "public",
        "interface", "recursion", "iteration", "conditional", "expression",
        "statement", "declaration", "memory", "cache", "buffer", "socket",
    ]
]  # 84 pairs per spec

FIRST_LETTER_PAIRS: List[Tuple[str, str]] = [
    (w, w[0]) for w in [
        "apple", "banana", "cherry", "dragon", "eagle", "falcon", "guitar",
        "hammer", "island", "jungle", "kitten", "lemon", "mango", "needle",
        "orange", "pencil", "quartz", "rabbit", "silver", "turtle", "umbrella",
        "violet", "window", "yellow", "zebra", "anchor", "bridge", "castle",
        "desert", "engine", "forest", "garden", "harbor", "insect", "jacket",
        "kernel", "lantern", "mirror", "nephew", "ocean", "palace", "quarter",
        "rocket", "sunset", "temple", "unique", "valley", "waffle", "xenon",
        "yogurt", "zipper", "basket", "candle", "dagger", "emerald", "feather",
        "goblet", "helmet", "igloo", "jersey", "kettle", "legend", "marble",
        "napkin", "oyster", "pebble", "ribbon", "saddle", "trophy", "voyage",
        "walnut", "magnet", "planet", "stream", "throne", "velvet", "winter",
        "button", "copper", "dinner", "finger", "gravel", "hollow", "justice",
        "leather", "matter",
    ]
]  # 86 pairs per spec

ENGLISH_SPANISH_PAIRS: List[Tuple[str, str]] = [
    ("cat", "gato"), ("dog", "perro"), ("house", "casa"), ("water", "agua"),
    ("book", "libro"), ("sun", "sol"), ("moon", "luna"), ("star", "estrella"),
    ("tree", "arbol"), ("flower", "flor"), ("fish", "pez"), ("bird", "pajaro"),
    ("stone", "piedra"), ("fire", "fuego"), ("earth", "tierra"),
    ("sky", "cielo"), ("rain", "lluvia"), ("snow", "nieve"), ("wind", "viento"),
    ("sea", "mar"), ("river", "rio"), ("mountain", "montana"),
    ("forest", "bosque"), ("island", "isla"), ("bridge", "puente"),
    ("door", "puerta"), ("window", "ventana"), ("table", "mesa"),
    ("chair", "silla"), ("bed", "cama"), ("mirror", "espejo"),
    ("clock", "reloj"), ("key", "llave"), ("shoe", "zapato"),
    ("hat", "sombrero"), ("shirt", "camisa"), ("bread", "pan"),
    ("milk", "leche"), ("egg", "huevo"), ("cheese", "queso"),
    ("sugar", "azucar"), ("salt", "sal"), ("meat", "carne"),
    ("fruit", "fruta"), ("rice", "arroz"), ("king", "rey"),
    ("queen", "reina"), ("prince", "principe"), ("gold", "oro"),
    ("silver", "plata"), ("iron", "hierro"), ("glass", "vidrio"),
    ("paper", "papel"), ("heart", "corazon"), ("hand", "mano"),
    ("head", "cabeza"), ("eye", "ojo"), ("mouth", "boca"),
    ("nose", "nariz"), ("tooth", "diente"), ("blood", "sangre"),
    ("bone", "hueso"), ("skin", "piel"), ("hair", "pelo"),
    ("friend", "amigo"), ("brother", "hermano"), ("sister", "hermana"),
    ("father", "padre"), ("mother", "madre"), ("son", "hijo"),
    ("daughter", "hija"), ("child", "nino"), ("man", "hombre"),
    ("woman", "mujer"), ("night", "noche"), ("day", "dia"),
    ("week", "semana"), ("month", "mes"), ("year", "ano"),
    ("time", "tiempo"), ("world", "mundo"), ("city", "ciudad"),
    ("street", "calle"), ("school", "escuela"), ("church", "iglesia"),
    ("war", "guerra"), ("peace", "paz"), ("love", "amor"),
]  # 88 pairs per spec

SENTIMENT_PAIRS: List[Tuple[str, str]] = [
    ("I love this product", "I hate this product"),
    ("Great experience overall", "Terrible experience overall"),
    ("Highly recommended", "Strongly not recommended"),
    ("Excellent quality", "Poor quality"),
    ("Very satisfied", "Very disappointed"),
    ("Amazing service", "Awful service"),
    ("Best purchase ever", "Worst purchase ever"),
    ("Wonderful experience", "Horrible experience"),
    ("Fantastic results", "Disastrous results"),
    ("Perfect condition", "Broken condition"),
    ("Outstanding performance", "Abysmal performance"),
    ("Delightful surprise", "Unpleasant surprise"),
    ("Exceptional value", "Overpriced junk"),
    ("Brilliant design", "Terrible design"),
    ("Superb craftsmanship", "Shoddy craftsmanship"),
    ("Absolutely fantastic", "Absolutely horrible"),
    ("Impressive quality", "Disappointing quality"),
    ("Remarkable achievement", "Complete failure"),
    ("Exceeded expectations", "Failed expectations"),
    ("Top notch service", "Bottom tier service"),
    ("Phenomenal results", "Pathetic results"),
    ("Incredible value", "Waste of money"),
    ("Stunning performance", "Dismal performance"),
    ("Marvelous experience", "Dreadful experience"),
    ("Flawless execution", "Botched execution"),
    ("Superior product", "Inferior product"),
    ("Magnificent work", "Sloppy work"),
    ("Splendid outcome", "Terrible outcome"),
    ("Glorious success", "Miserable failure"),
    ("Great movie", "Terrible movie"),
    ("Wonderful show", "Awful show"),
    ("Excellent book", "Dreadful book"),
    ("Amazing food", "Disgusting food"),
    ("Perfect weather", "Horrible weather"),
    ("Beautiful scenery", "Ugly scenery"),
    ("Lovely atmosphere", "Unpleasant atmosphere"),
    ("Friendly staff", "Rude staff"),
    ("Clean environment", "Dirty environment"),
    ("Fast delivery", "Slow delivery"),
    ("Smooth transaction", "Messy transaction"),
    ("Reliable service", "Unreliable service"),
    ("Enjoyable trip", "Miserable trip"),
    ("Productive meeting", "Wasteful meeting"),
    ("Exciting adventure", "Boring adventure"),
    ("Comfortable stay", "Uncomfortable stay"),
    ("Tasty meal", "Bland meal"),
    ("Refreshing drink", "Disgusting drink"),
    ("Cozy room", "Cold room"),
    ("Charming town", "Depressing town"),
    ("Scenic route", "Ugly route"),
    ("Warm welcome", "Cold reception"),
    ("Kind gesture", "Rude gesture"),
    ("Fair price", "Unfair price"),
    ("Clear instructions", "Confusing instructions"),
    ("Helpful advice", "Useless advice"),
    ("Strong argument", "Weak argument"),
    ("Clever solution", "Stupid solution"),
    ("Bright future", "Dark future"),
    ("Fresh start", "Bad start"),
    ("Good news", "Bad news"),
]

REVERSE_WORD_PAIRS: List[Tuple[str, str]] = [
    (w, w[::-1]) for w in [
        "hello", "world", "python", "table", "chair", "stone", "plant",
        "river", "cloud", "flame", "ocean", "tiger", "lemon", "grape",
        "pearl", "storm", "frost", "blaze", "crane", "eagle", "shark",
        "whale", "coral", "maple", "cedar", "bloom", "trail", "grove",
        "brook", "ridge", "cliff", "shore", "delta", "marsh", "plain",
        "grain", "wheat", "steel", "brick", "glass", "metal", "ivory",
        "amber", "opal", "ruby", "jade", "onyx", "slate", "chalk",
        "quilt", "scarf", "badge", "arrow", "blade", "crown", "tower",
        "vault", "forge", "anvil", "prism", "pixel", "glyph", "nexus",
        "crypt", "lunar", "solar", "vapor", "birch", "plume", "orbit",
        "pulse", "gleam", "spark", "ember", "flint", "spear", "lance",
        "sword", "mango", "peach",
    ]
]

OBJECT_COLOR_PAIRS: List[Tuple[str, str]] = [
    ("banana", "yellow"), ("grass", "green"), ("sky", "blue"),
    ("snow", "white"), ("tomato", "red"), ("coal", "black"),
    ("carrot", "orange"), ("eggplant", "purple"), ("lemon", "yellow"),
    ("strawberry", "red"), ("blueberry", "blue"), ("lime", "green"),
    ("pumpkin", "orange"), ("cherry", "red"), ("lettuce", "green"),
    ("milk", "white"), ("chocolate", "brown"), ("gold", "gold"),
    ("silver", "silver"), ("ruby", "red"), ("emerald", "green"),
    ("sapphire", "blue"), ("pearl", "white"), ("amber", "orange"),
    ("ivory", "white"), ("charcoal", "black"), ("lavender", "purple"),
    ("salmon", "pink"), ("flamingo", "pink"), ("raven", "black"),
    ("dove", "white"), ("crow", "black"), ("canary", "yellow"),
    ("cardinal", "red"), ("parrot", "green"), ("swan", "white"),
    ("panther", "black"), ("polar bear", "white"), ("lobster", "red"),
    ("sunflower", "yellow"), ("rose", "red"), ("violet", "purple"),
    ("daisy", "white"), ("tulip", "red"), ("dandelion", "yellow"),
    ("marigold", "orange"), ("iris", "purple"), ("lily", "white"),
    ("fire truck", "red"), ("school bus", "yellow"), ("taxi", "yellow"),
    ("stop sign", "red"), ("cloud", "white"), ("midnight", "black"),
    ("sunset", "orange"), ("ocean", "blue"), ("forest", "green"),
    ("sand", "tan"), ("brick", "red"), ("cotton", "white"),
    ("ink", "black"), ("butter", "yellow"), ("honey", "golden"),
    ("wine", "red"), ("coffee", "brown"), ("tea", "brown"),
    ("cream", "white"), ("rust", "orange"), ("copper", "brown"),
    ("jade", "green"), ("onyx", "black"), ("opal", "white"),
    ("topaz", "yellow"), ("garnet", "red"), ("turquoise", "blue"),
    ("amethyst", "purple"), ("obsidian", "black"), ("marble", "white"),
    ("slate", "gray"), ("granite", "gray"), ("chalk", "white"),
    ("ash", "gray"), ("moss", "green"), ("bark", "brown"),
    ("bamboo", "green"),
]  # 85 pairs per spec

# Alternative valid outputs for object_color
OBJECT_COLOR_ALTERNATIVES: Dict[str, List[str]] = {
    "rose": ["red", "pink", "white", "yellow"],
    "tulip": ["red", "pink", "yellow", "white", "purple"],
    "grape": ["purple", "green", "red"],
    "apple": ["red", "green", "yellow"],
    "pepper": ["black", "red", "green", "yellow"],
    "tea": ["brown", "green", "black"],
    "wine": ["red", "white", "pink"],
    "marble": ["white", "gray", "black"],
}


# ===========================================================================
# Task definitions — 8 templates per task (T1-T8), all task-specific
# ===========================================================================

def build_task_registry() -> Dict[str, TaskSpec]:
    """Build the complete registry of 12 tasks per EXPERIMENT_REDESIGN_SPEC.md."""

    registry: Dict[str, TaskSpec] = {}

    # -----------------------------------------------------------------------
    # 1. LEXICAL RETRIEVAL
    # -----------------------------------------------------------------------

    registry["antonym"] = TaskSpec(
        name="antonym",
        category=TaskCategory.LEXICAL_RETRIEVAL,
        description="Given an adjective, produce its antonym",
        expected_difficulty="easy",
        templates={
            "T1": "The opposite of {X} is",
            "T2": "{X} has the opposite meaning of",
            "T3": "antonym({X}) =",
            "T4": "opposite: {X} -->",
            "T5": "What is the antonym of {X}?",
            "T6": "What word means the opposite of {X}?",
            "T7": "Antonym relation: {X} maps to",
            "T8": "Given the word {X}, the antonym is",
        },
        pairs=ANTONYM_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
    )

    registry["synonym"] = TaskSpec(
        name="synonym",
        category=TaskCategory.LEXICAL_RETRIEVAL,
        description="Given a word, produce a synonym",
        expected_difficulty="medium",
        templates={
            "T1": "A synonym for {X} is",
            "T2": "Another word for {X} is",
            "T3": "synonym({X}) =",
            "T4": "similar_word: {X} -->",
            "T5": "What is a synonym of {X}?",
            "T6": "What word has a similar meaning to {X}?",
            "T7": "Synonym identification: {X} corresponds to",
            "T8": "The word {X} is synonymous with",
        },
        pairs=SYNONYM_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
        alternative_outputs=SYNONYM_ALTERNATIVES,
    )

    registry["hypernym"] = TaskSpec(
        name="hypernym",
        category=TaskCategory.LEXICAL_RETRIEVAL,
        description="Given a word, produce its category (hypernym)",
        expected_difficulty="medium",
        templates={
            "T1": "{X} is a type of",
            "T2": "The category of {X} is",
            "T3": "hypernym({X}) =",
            "T4": "category: {X} -->",
            "T5": "What kind of thing is a {X}?",
            "T6": "What category does {X} belong to?",
            "T7": "Taxonomic classification: {X} is classified as",
            "T8": "In terms of hierarchy, {X} falls under",
        },
        pairs=HYPERNYM_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
        alternative_outputs=HYPERNYM_ALTERNATIVES,
    )

    # -----------------------------------------------------------------------
    # 2. FACTUAL RETRIEVAL
    # -----------------------------------------------------------------------

    registry["country_capital"] = TaskSpec(
        name="country_capital",
        category=TaskCategory.FACTUAL_RETRIEVAL,
        description="Given a country, produce its capital city",
        expected_difficulty="easy",
        templates={
            "T1": "The capital of {X} is",
            "T2": "{X}'s capital city is",
            "T3": "capital({X}) =",
            "T4": "country_capital: {X} -->",
            "T5": "What is the capital of {X}?",
            "T6": "Which city is the capital of {X}?",
            "T7": "Capital city identification: {X} has capital",
            "T8": "For the country {X}, the capital is",
        },
        pairs=COUNTRY_CAPITAL_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
    )

    registry["english_spanish"] = TaskSpec(
        name="english_spanish",
        category=TaskCategory.FACTUAL_RETRIEVAL,
        description="Translate an English word to Spanish",
        expected_difficulty="medium",
        templates={
            "T1": "The Spanish word for {X} is",
            "T2": "In Spanish, {X} is called",
            "T3": "translate_es({X}) =",
            "T4": "en_to_es: {X} -->",
            "T5": "How do you say {X} in Spanish?",
            "T6": "What is the Spanish translation of {X}?",
            "T7": "English-Spanish translation: {X} renders as",
            "T8": "The Spanish equivalent of {X} is",
        },
        pairs=ENGLISH_SPANISH_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
    )

    registry["object_color"] = TaskSpec(
        name="object_color",
        category=TaskCategory.FACTUAL_RETRIEVAL,
        description="Given a common object, produce its typical color",
        expected_difficulty="easy-medium",
        templates={
            "T1": "The color of a {X} is",
            "T2": "A {X} is typically colored",
            "T3": "color({X}) =",
            "T4": "object_color: {X} -->",
            "T5": "What color is a {X}?",
            "T6": "What is the typical color of a {X}?",
            "T7": "Color association: {X} corresponds to",
            "T8": "The characteristic color of {X} is",
        },
        pairs=OBJECT_COLOR_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
        alternative_outputs=OBJECT_COLOR_ALTERNATIVES,
    )

    # -----------------------------------------------------------------------
    # 3. MORPHOLOGICAL TRANSFORM
    # -----------------------------------------------------------------------

    registry["past_tense"] = TaskSpec(
        name="past_tense",
        category=TaskCategory.MORPHOLOGICAL_TRANSFORM,
        description="Convert a regular verb to past tense",
        expected_difficulty="easy",
        templates={
            "T1": "The past tense of {X} is",
            "T2": "Yesterday I {X}, so I",
            "T3": "past_tense({X}) =",
            "T4": "verb_past: {X} -->",
            "T5": "What is the past tense of {X}?",
            "T6": "How do you conjugate {X} in the past?",
            "T7": "Past tense conjugation: {X} becomes",
            "T8": "The simple past form of {X} is",
        },
        pairs=PAST_TENSE_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
    )

    registry["plural"] = TaskSpec(
        name="plural",
        category=TaskCategory.MORPHOLOGICAL_TRANSFORM,
        description="Convert a noun to its plural form",
        expected_difficulty="easy-medium",
        templates={
            "T1": "The plural of {X} is",
            "T2": "{X} in plural form is",
            "T3": "plural({X}) =",
            "T4": "noun_plural: {X} -->",
            "T5": "What is the plural of {X}?",
            "T6": "How do you pluralize {X}?",
            "T7": "Plural formation: {X} becomes",
            "T8": "The plural form of the noun {X} is",
        },
        pairs=PLURAL_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
    )

    # -----------------------------------------------------------------------
    # 4. CHARACTER / SURFACE
    # -----------------------------------------------------------------------

    registry["capitalize"] = TaskSpec(
        name="capitalize",
        category=TaskCategory.CHARACTER_SURFACE,
        description="Convert a lowercase word to uppercase",
        expected_difficulty="hard",
        templates={
            "T1": "{X} in uppercase is",
            "T2": "The uppercase version of {X} is",
            "T3": "UPPERCASE({X}) =",
            "T4": "to_upper: {X} -->",
            "T5": "What is {X} in all capital letters?",
            "T6": "How do you write {X} in uppercase?",
            "T7": "Uppercase conversion: {X} becomes",
            "T8": "Applying capitalization to {X} yields",
        },
        pairs=CAPITALIZE_WORDS,
        accuracy_mode=AccuracyMode.CASE_SENSITIVE_SUB,
    )

    registry["first_letter"] = TaskSpec(
        name="first_letter",
        category=TaskCategory.CHARACTER_SURFACE,
        description="Extract the first letter of a word",
        expected_difficulty="hard",
        max_new_tokens=3,
        templates={
            "T1": "The first letter of {X} is",
            "T2": "{X} starts with the letter",
            "T3": "first_char({X}) =",
            "T4": "initial: {X} -->",
            "T5": "What letter does {X} start with?",
            "T6": "What is the first letter of {X}?",
            "T7": "Initial letter extraction: {X} yields",
            "T8": "The leading character of {X} is",
        },
        pairs=FIRST_LETTER_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
    )

    registry["reverse_word"] = TaskSpec(
        name="reverse_word",
        category=TaskCategory.CHARACTER_SURFACE,
        description="Reverse the characters of a word",
        expected_difficulty="hard",
        templates={
            "T1": "{X} spelled backwards is",
            "T2": "The reverse of {X} is",
            "T3": "reverse({X}) =",
            "T4": "reversed: {X} -->",
            "T5": "What is {X} spelled in reverse?",
            "T6": "How do you spell {X} backwards?",
            "T7": "String reversal: {X} becomes",
            "T8": "Reversing the characters of {X} yields",
        },
        pairs=REVERSE_WORD_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
    )

    # -----------------------------------------------------------------------
    # 5. COMPOSITIONAL / SEMANTIC
    # -----------------------------------------------------------------------

    registry["sentiment_flip"] = TaskSpec(
        name="sentiment_flip",
        category=TaskCategory.COMPOSITIONAL_SEMANTIC,
        description="Rewrite a positive phrase with negative sentiment",
        expected_difficulty="hard",
        max_new_tokens=10,
        templates={
            "T1": "Rewrite with opposite sentiment: {X} becomes",
            "T2": "The negative version of {X} is",
            "T3": "flip_sentiment({X}) =",
            "T4": "sentiment_reverse: {X} -->",
            "T5": "What is the opposite sentiment of {X}?",
            "T6": "How would you express {X} negatively?",
            "T7": "Sentiment inversion: {X} transforms to",
            "T8": "Applying sentiment reversal to {X} yields",
        },
        pairs=SENTIMENT_PAIRS,
        accuracy_mode=AccuracyMode.SUBSTRING,
    )

    return registry


# Singleton
TASK_REGISTRY: Dict[str, TaskSpec] = build_task_registry()


def get_task(name: str) -> TaskSpec:
    """Get a task by name, raising a clear error if not found."""
    if name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]


def get_tasks(names: Optional[List[str]] = None) -> Dict[str, TaskSpec]:
    """Get multiple tasks by name (None = all)."""
    if names is None:
        return dict(TASK_REGISTRY)
    return {n: get_task(n) for n in names}


def validate_all_tasks() -> Dict[str, List[str]]:
    """Validate all tasks and return {task_name: [warnings]}."""
    results = {}
    for name, spec in TASK_REGISTRY.items():
        warnings = spec.validate()
        results[name] = warnings
        for w in warnings:
            logger.warning("Task '%s': %s", name, w)
    return results


def print_task_summary():
    """Print a human-readable summary of all tasks."""
    print(f"{'Task':<20} {'Category':<26} {'Difficulty':<12} {'Pairs':<6} {'Templates':<10} {'Eval'}")
    print("-" * 95)
    for name, spec in TASK_REGISTRY.items():
        print(
            f"{name:<20} {spec.category.value:<26} {spec.expected_difficulty:<12} "
            f"{spec.n_pairs:<6} {spec.n_templates:<10} {spec.accuracy_mode.value}"
        )


if __name__ == "__main__":
    print_task_summary()
    print()
    warnings = validate_all_tasks()
    total_w = sum(len(w) for w in warnings.values())
    print(f"\nValidation: {total_w} warnings across {len(TASK_REGISTRY)} tasks")
