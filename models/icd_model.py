from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ICDCode:
    value: str

@dataclass
class Term:
    title: str
    level: Optional[str]
    codes: List[str]
    see: List[str]
    see_also: List[str]
    modifiers: List[str]
    subterms: List['Term']

@dataclass
class Letter:
    name: str
    terms: List[Term]

@dataclass
class ICDIndex:
    version: str
    title: str
    letters: List[Letter]