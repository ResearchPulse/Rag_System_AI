"""Knowledge Graph Store interface for GraphRAG & entity relationship tracking."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class GraphNode(BaseModel):
    id: str
    label: str
    properties: Dict[str, Any] = {}


class GraphRelation(BaseModel):
    from_node_id: str
    relation_type: str
    to_node_id: str
    properties: Dict[str, Any] = {}


class GraphQueryResult(BaseModel):
    nodes: List[GraphNode] = []
    relations: List[GraphRelation] = []
    raw_records: List[Dict[str, Any]] = []


class BaseGraphStore(ABC):
    """Abstract interface for Knowledge Graph Store (Neo4j)."""

    @abstractmethod
    async def add_node(self, label: str, node_id: str, properties: Dict[str, Any]) -> bool:
        pass

    @abstractmethod
    async def add_relation(
        self,
        from_node_id: str,
        relation_type: str,
        to_node_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        pass

    @abstractmethod
    async def query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> GraphQueryResult:
        pass
