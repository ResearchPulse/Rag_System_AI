"""Knowledge Graph Retriever using Neo4j.

Executes domain Cypher queries to extract graph relationships (co-authorship,
topic clusters, journal indexing, citations) and converts them into structured chunks.
"""

import logging
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.modules.indexing.graph_store.neo4j_store import Neo4jGraphStore
from app.modules.retrieval.schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class GraphRetriever:
    """Retrieves relational context from Neo4j Knowledge Graph."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.store = Neo4jGraphStore()

    def retrieve(
        self,
        query: str,
        entities: Dict[str, Any],
        limit: int = 5,
    ) -> List[RetrievedChunk]:
        """Dispatches graph search based on entities extracted from the query."""
        driver = self.store.get_driver()
        if driver is None:
            logger.warning("Neo4j driver is not available.")
            return []

        chunks: List[RetrievedChunk] = []

        authors = entities.get("authors", [])
        topics = entities.get("topics", [])
        journals = entities.get("journals", [])

        # Case 1: Search by Author
        if authors:
            for author_name in authors:
                chunks.extend(self._search_by_author(driver, author_name, limit))

        # Case 2: Search by Topic
        elif topics:
            for topic_name in topics:
                chunks.extend(self._search_by_topic(driver, topic_name, limit))

        # Case 3: Search by Journal
        elif journals:
            for journal_name in journals:
                chunks.extend(self._search_by_journal(driver, journal_name, limit))

        # Case 4: General keyword search on graph
        else:
            chunks.extend(self._search_general_graph(driver, query, limit))

        return chunks[:limit]

    def _search_by_author(self, driver: Any, author_name: str, limit: int) -> List[RetrievedChunk]:
        cypher = """
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($name)
            OPTIONAL MATCH (a)-[:WRITES]->(art:Article)
            OPTIONAL MATCH (a)-[:COLLABORATES_WITH]-(co:Author)
            WITH a, 
                 collect(DISTINCT art.title)[..10] AS articles,
                 collect(DISTINCT co.name)[..10] AS collaborators
            RETURN a.name AS author_name,
                   articles,
                   collaborators
            LIMIT $limit
        """
        chunks = []
        try:
            with driver.session(database=self.settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, name=author_name, limit=limit)
                for record in result:
                    name = record["author_name"]
                    arts = record["articles"]
                    collabs = record["collaborators"]

                    collab_str = ", ".join(collabs) if collabs else "Không có dữ liệu đồng tác giả"
                    art_str = "; ".join(arts[:5]) if arts else "Chưa có bài báo liên kết"

                    content = (
                        f"[Knowledge Graph - Tác giả] {name}.\n"
                        f"- Các đồng tác giả thường hợp tác: {collab_str}.\n"
                        f"- Một số bài báo tiêu biểu: {art_str}."
                    )
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=f"graph_author_{name.replace(' ', '_')}",
                            document_id=f"doc_author_{name.replace(' ', '_')}",
                            content=content,
                            score=0.98,
                            metadata={
                                "source": "neo4j",
                                "type": "AuthorGraph",
                                "author": name,
                                "collaborators_count": len(collabs),
                                "articles_count": len(arts),
                            },
                        )
                    )
        except Exception as e:
            logger.error(f"Error querying author graph in Neo4j: {e}")

        return chunks

    def _search_by_topic(self, driver: Any, topic_name: str, limit: int) -> List[RetrievedChunk]:
        cypher = """
            MATCH (t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($topic)
            MATCH (art:Article)-[:HAS_TOPIC]->(t)
            OPTIONAL MATCH (a:Author)-[:WRITES]->(art)
            WITH t, art, collect(DISTINCT a.name)[..3] AS authors
            RETURN t.name AS topic_name,
                   art.title AS title,
                   art.publication_year AS year,
                   authors
            ORDER BY art.publication_year DESC
            LIMIT $limit
        """
        chunks = []
        try:
            with driver.session(database=self.settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, topic=topic_name, limit=limit)
                for record in result:
                    t_name = record["topic_name"]
                    title = record["title"]
                    year = record["year"]
                    authors = ", ".join(record["authors"]) or "N/A"

                    content = (
                        f"[Knowledge Graph - Chủ đề: {t_name}]\n"
                        f"- Bài báo: '{title}' ({year}).\n"
                        f"- Tác giả: {authors}."
                    )
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=f"graph_topic_{t_name[:15]}",
                            document_id=f"doc_topic_{t_name[:15]}",
                            content=content,
                            score=0.95,
                            metadata={"source": "neo4j", "type": "TopicGraph", "topic": t_name, "title": title, "year": year},
                        )
                    )
        except Exception as e:
            logger.error(f"Error querying topic graph in Neo4j: {e}")

        return chunks

    def _search_by_journal(self, driver: Any, journal_name: str, limit: int) -> List[RetrievedChunk]:
        cypher = """
            MATCH (j:Journal)
            WHERE toLower(j.name) CONTAINS toLower($journal)
            MATCH (art:Article)-[:PUBLISHED_IN]->(j)
            RETURN j.name AS journal_name,
                   j.issn AS issn,
                   art.title AS title,
                   art.publication_year AS year
            ORDER BY art.publication_year DESC
            LIMIT $limit
        """
        chunks = []
        try:
            with driver.session(database=self.settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, journal=journal_name, limit=limit)
                for record in result:
                    j_name = record["journal_name"]
                    title = record["title"]
                    year = record["year"]
                    issn = record.get("issn", "")

                    content = (
                        f"[Knowledge Graph - Tạp chí: {j_name} (ISSN: {issn})]\n"
                        f"- Bài báo đã xuất bản: '{title}' ({year})."
                    )
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=f"graph_journal_{j_name[:15]}",
                            document_id=f"doc_journal_{j_name[:15]}",
                            content=content,
                            score=0.93,
                            metadata={"source": "neo4j", "type": "JournalGraph", "journal": j_name, "title": title, "year": year},
                        )
                    )
        except Exception as e:
            logger.error(f"Error querying journal graph in Neo4j: {e}")

        return chunks

    def _search_general_graph(self, driver: Any, query: str, limit: int) -> List[RetrievedChunk]:
        # Search across Authors, Topics, and Articles titles
        cypher = """
            MATCH (a:Author)
            WHERE toLower(a.name) CONTAINS toLower($term)
            OPTIONAL MATCH (a)-[:COLLABORATES_WITH]-(co:Author)
            RETURN a.name AS name, 'Author' AS kind, collect(DISTINCT co.name)[..5] AS details
            LIMIT $limit
            UNION
            MATCH (t:Topic)
            WHERE toLower(t.name) CONTAINS toLower($term)
            OPTIONAL MATCH (art:Article)-[:HAS_TOPIC]->(t)
            RETURN t.name AS name, 'Topic' AS kind, collect(DISTINCT art.title)[..5] AS details
            LIMIT $limit
        """
        chunks = []
        try:
            with driver.session(database=self.settings.NEO4J_DATABASE) as session:
                result = session.run(cypher, term=query.strip(), limit=limit)
                for record in result:
                    name = record["name"]
                    kind = record["kind"]
                    details = ", ".join(record["details"] or [])
                    content = f"[Knowledge Graph - {kind}] {name}. Liên kết: {details}."
                    chunks.append(
                        RetrievedChunk(
                            chunk_id=f"graph_gen_{name[:15]}",
                            document_id=f"doc_gen_{name[:15]}",
                            content=content,
                            score=0.90,
                            metadata={"source": "neo4j", "type": kind, "name": name},
                        )
                    )
        except Exception as e:
            logger.error(f"Error querying general graph in Neo4j: {e}")

        return chunks
