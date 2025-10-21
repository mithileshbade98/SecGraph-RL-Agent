"""
Graph loader with bitemporal Neo4j support.

Loads synthetic security events into a temporal identity graph with:
- Nodes: Person, Email, Account, Device, IP, Session, Card, Resource, Tenant
- Edges: USES, LOGS_IN_FROM, SAME_DEVICE_AS, etc. with temporal attributes
- Bitemporal tracking: valid_from, valid_to, observed_at
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from neo4j import GraphDatabase, Driver
import pandas as pd
from pathlib import Path
import yaml
from loguru import logger


class BiTemporalGraphLoader:
    """Load events into Neo4j with bitemporal tracking."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """Initialize Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database
        logger.info(f"Connected to Neo4j at {uri}")

    def close(self):
        """Close Neo4j connection."""
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def create_constraints(self):
        """Create uniqueness constraints and indexes."""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Email) REQUIRE e.address IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Device) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:IP) REQUIRE i.address IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Session) REQUIRE s.id IS UNIQUE",
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS FOR (u:User) ON (u.created_at)",
            "CREATE INDEX IF NOT EXISTS FOR (d:Device) ON (d.first_seen)",
            "CREATE INDEX IF NOT EXISTS FOR (i:IP) ON (i.first_seen)",
            "CREATE INDEX IF NOT EXISTS FOR (s:Session) ON (s.timestamp)",
        ]

        with self.driver.session(database=self.database) as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.debug(f"Created constraint: {constraint}")
                except Exception as e:
                    logger.warning(f"Constraint already exists or error: {e}")

            for index in indexes:
                try:
                    session.run(index)
                    logger.debug(f"Created index: {index}")
                except Exception as e:
                    logger.warning(f"Index already exists or error: {e}")

    def upsert_node(self, tx, label: str, properties: Dict[str, Any], id_key: str = "id"):
        """Upsert a node with bitemporal attributes."""
        query = f"""
        MERGE (n:{label} {{{id_key}: $id}})
        SET n += $properties
        RETURN n
        """
        result = tx.run(query, id=properties[id_key], properties=properties)
        return result.single()

    def create_temporal_edge(
        self,
        tx,
        from_label: str,
        from_id: str,
        to_label: str,
        to_id: str,
        rel_type: str,
        properties: Dict[str, Any],
        valid_from: datetime,
        valid_to: Optional[datetime] = None,
        observed_at: Optional[datetime] = None,
    ):
        """Create edge with bitemporal attributes."""
        if observed_at is None:
            observed_at = datetime.now()

        # Convert datetimes to ISO strings
        temporal_props = {
            "valid_from": valid_from.isoformat(),
            "valid_to": valid_to.isoformat() if valid_to else None,
            "observed_at": observed_at.isoformat(),
            **properties,
        }

        query = f"""
        MATCH (from:{from_label} {{id: $from_id}})
        MATCH (to:{to_label} {{id: $to_id}})
        CREATE (from)-[r:{rel_type}]->(to)
        SET r += $properties
        RETURN r
        """
        result = tx.run(query, from_id=from_id, to_id=to_id, properties=temporal_props)
        return result.single()

    def load_event(self, event: Dict[str, Any]):
        """Load a single security event into the graph."""
        timestamp = event['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        with self.driver.session(database=self.database) as session:
            # Create User node
            user_node = {
                'id': event['user_id'],
                'created_at': timestamp.isoformat(),
                'anomaly_type': event.get('anomaly_type', 'normal'),
            }
            session.execute_write(self.upsert_node, "User", user_node)

            # Create Email node
            email_node = {
                'id': event['email'],
                'address': event['email'],
                'first_seen': timestamp.isoformat(),
            }
            session.execute_write(self.upsert_node, "Email", email_node, id_key='address')

            # Create Device node
            device_node = {
                'id': event['device_id'],
                'first_seen': timestamp.isoformat(),
            }
            session.execute_write(self.upsert_node, "Device", device_node)

            # Create IP node
            ip_node = {
                'id': event['ip_address'],
                'address': event['ip_address'],
                'first_seen': timestamp.isoformat(),
                'geo_location': event.get('geo_location', 'Unknown'),
            }
            session.execute_write(self.upsert_node, "IP", ip_node, id_key='address')

            # Create Session node
            session_node = {
                'id': event['session_id'],
                'timestamp': timestamp.isoformat(),
                'event_type': event['event_type'],
            }
            session.execute_write(self.upsert_node, "Session", session_node)

            # Create relationships with temporal attributes
            # User -> Email
            session.execute_write(
                self.create_temporal_edge,
                "User", event['user_id'],
                "Email", event['email'],
                "USES",
                {},
                valid_from=timestamp,
                observed_at=timestamp,
            )

            # User -> Device
            session.execute_write(
                self.create_temporal_edge,
                "User", event['user_id'],
                "Device", event['device_id'],
                "USES",
                {},
                valid_from=timestamp,
                observed_at=timestamp,
            )

            # Session -> IP
            session.execute_write(
                self.create_temporal_edge,
                "Session", event['session_id'],
                "IP", event['ip_address'],
                "FROM_IP",
                {},
                valid_from=timestamp,
                observed_at=timestamp,
            )

            # User -> Session
            session.execute_write(
                self.create_temporal_edge,
                "User", event['user_id'],
                "Session", event['session_id'],
                "HAS_SESSION",
                {},
                valid_from=timestamp,
                observed_at=timestamp,
            )

    def load_from_parquet(self, parquet_file: Path, batch_size: int = 1000):
        """Load events from parquet file."""
        df = pd.read_parquet(parquet_file)
        logger.info(f"Loading {len(df)} events from {parquet_file}")

        # Create constraints first
        self.create_constraints()

        # Load events in batches
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            for _, event in batch.iterrows():
                try:
                    self.load_event(event.to_dict())
                except Exception as e:
                    logger.error(f"Error loading event: {e}")
                    continue

            logger.info(f"Loaded {min(i + batch_size, len(df))}/{len(df)} events")

        logger.info("Graph loading complete!")

    def get_as_of_snapshot(self, as_of: datetime, limit: int = 100) -> List[Dict[str, Any]]:
        """Query graph as-of a specific time (bitemporal query)."""
        query = """
        MATCH (u:User)-[r]->(n)
        WHERE datetime(r.valid_from) <= $as_of
        AND (r.valid_to IS NULL OR datetime(r.valid_to) > $as_of)
        RETURN u, r, n
        LIMIT $limit
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(
                query,
                as_of=as_of.isoformat(),
                limit=limit
            )
            return [record.data() for record in result]

    def detect_shared_device_clusters(self, min_accounts: int = 3) -> List[Dict[str, Any]]:
        """Detect clusters of accounts sharing the same device."""
        query = """
        MATCH (u1:User)-[:USES]->(d:Device)<-[:USES]-(u2:User)
        WHERE u1.id < u2.id
        WITH d, collect(DISTINCT u1.id) + collect(DISTINCT u2.id) AS users
        WHERE size(users) >= $min_accounts
        RETURN d.id AS device_id, users, size(users) AS account_count
        ORDER BY account_count DESC
        """
        with self.driver.session(database=self.database) as session:
            result = session.run(query, min_accounts=min_accounts)
            return [record.data() for record in result]

    def get_graph_stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        with self.driver.session(database=self.database) as session:
            # Count nodes by type
            node_query = """
            MATCH (n)
            RETURN labels(n)[0] AS label, count(n) AS count
            """
            node_result = session.run(node_query)
            nodes = {record['label']: record['count'] for record in node_result}

            # Count relationships by type
            rel_query = """
            MATCH ()-[r]->()
            RETURN type(r) AS type, count(r) AS count
            """
            rel_result = session.run(rel_query)
            rels = {record['type']: record['count'] for record in rel_result}

            return {'nodes': nodes, 'relationships': rels}


def main():
    """CLI entry point for graph loading."""
    import argparse
    import os
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Load synthetic data into Neo4j")
    parser.add_argument("--input", default="data/synthetic/security_events.parquet",
                        help="Input parquet file")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                        help="Neo4j URI")
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"),
                        help="Neo4j username")
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "secgraph123"),
                        help="Neo4j password")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    args = parser.parse_args()

    with BiTemporalGraphLoader(args.uri, args.user, args.password) as loader:
        # Load data
        loader.load_from_parquet(Path(args.input), batch_size=args.batch_size)

        # Print stats
        stats = loader.get_graph_stats()
        logger.info(f"Graph statistics: {stats}")

        # Detect shared device clusters
        clusters = loader.detect_shared_device_clusters(min_accounts=3)
        logger.info(f"Found {len(clusters)} shared device clusters")
        for cluster in clusters[:5]:
            logger.info(f"  Device {cluster['device_id']}: {cluster['account_count']} accounts")


if __name__ == "__main__":
    main()
