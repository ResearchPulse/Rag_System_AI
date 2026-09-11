"""Neo4j Knowledge Graph Store implementation."""
from typing import Any, Dict, List, Optional
import logging
try:
    from neo4j import GraphDatabase  # type: ignore
except ImportError:  # pragma: no cover
    GraphDatabase = None  # type: ignore

from app.core.config import get_settings
from app.modules.indexing.graph_store.base import BaseGraphStore, GraphNode, GraphRelation, GraphQueryResult

logger = logging.getLogger(__name__)


class Neo4jGraphStore(BaseGraphStore):
    """Neo4j Driver adapter for Knowledge Graph operations."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._driver: Any = None

    def get_driver(self) -> Any:
        """Lazy initialization of Neo4j Async/Sync driver."""
        if GraphDatabase is None:
            return None

        if self._driver is None:
            try:
                self._driver = GraphDatabase.driver(
                    self.settings.NEO4J_URI,
                    auth=(self.settings.NEO4J_USER, self.settings.NEO4J_PASSWORD),
                )
                logger.info(f"Connected to Neo4j at {self.settings.NEO4J_URI}")
            except Exception as e:
                logger.warning(f"Could not connect to Neo4j: {e}")
                self._driver = None
        return self._driver

    async def add_node(self, label: str, node_id: str, properties: Dict[str, Any]) -> bool:
        driver = self.get_driver()
        if driver is None:
            logger.info(f"[Mock/Fallback] Added node ({label} {{id: '{node_id}'}})")
            return True

        cypher = f"""
            MERGE (n:{label} {{id: $node_id}})
            SET n += $props
            RETURN n
        """
        try:
            with driver.session(database=self.settings.NEO4J_DATABASE) as session:
                session.run(cypher, node_id=node_id, props=properties)
            return True
        except Exception as e:
            logger.error(f"Error adding Neo4j node {label}:{node_id} - {e}")
            return False

    async def add_relation(
        self,
        from_node_id: str,
        relation_type: str,
        to_node_id: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        driver = self.get_driver()
        if driver is None:
            logger.info(f"[Mock/Fallback] Added relation (:{from_node_id})-[:{relation_type}]->(:{to_node_id})")
            return True

        props = properties or {}
        cypher = f"""
            MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
            MERGE (a)-[r:{relation_type}]->(b)
            SET r += $props
            RETURN r
        """
        try:
            with driver.session(database=self.settings.NEO4J_DATABASE) as session:
                session.run(cypher, from_id=from_node_id, to_id=to_node_id, props=props)
            return True
        except Exception as e:
            logger.error(f"Error creating Neo4j relation {relation_type} - {e}")
            return False

    async def query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> GraphQueryResult:
        driver = self.get_driver()
        if driver is None:
            return GraphQueryResult(
                nodes=[
                    GraphNode(id="art_1", label="Article", properties={"title": "RAG in Scientific Trends"}),
                    GraphNode(id="topic_1", label="Topic", properties={"name": "Artificial Intelligence"}),
                ],
                relations=[
                    GraphRelation(from_node_id="art_1", relation_type="BELONGS_TO", to_node_id="topic_1")
                ],
                raw_records=[],
            )

        params = parameters or {}
        try:
            with driver.session(database=self.settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, **params)
                records = [record.data() for record in result]
                return GraphQueryResult(raw_records=records)
        except Exception as e:
            logger.error(f"Error querying Neo4j: {e}")
            return GraphQueryResult()

    def close(self):
        if self._driver:
            self._driver.close()
            self._driver = None
