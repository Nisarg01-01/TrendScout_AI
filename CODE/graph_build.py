import os
import json
import pandas as pd
from neo4j import GraphDatabase
import config
from ast import literal_eval
import networkx as nx
from datetime import datetime
from collections import defaultdict
import numpy as np
import hashlib
import argparse

class GraphBuilder:
    def __init__(self):
        if not config.NEO4J_URI or str(config.NEO4J_URI).strip() == "":
            raise ValueError(
                "NEO4J_URI is missing/blank. Set it in `.env` (repo root) "
                "to something like `bolt://localhost:7687` or `neo4j+s://<id>.databases.neo4j.io`."
            )
        if "://" not in str(config.NEO4J_URI):
            raise ValueError(f"NEO4J_URI looks invalid: {config.NEO4J_URI!r}")
        self.driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD)
        )
        self.verify_connection()

    def verify_connection(self):
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS num")
                print(f"Connected to Neo4j: {result.single()['num'] == 1}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Dangerous: deletes all nodes/relationships from the configured Neo4j database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        """Create graph database constraints and indexes for unique nodes and performance."""
        
        # Drop incorrect KPI constraint on 'type' property if exists
        with self.driver.session() as session:
            try:
                result = session.run("SHOW CONSTRAINTS")
                for record in result:
                    record_dict = dict(record)
                    constraint_name = record_dict.get('name', '')
                    labels = record_dict.get('labelsOrTypes', [])
                    properties = record_dict.get('properties', [])
                    
                    # Drop KPI constraint on 'type' property (incorrect)
                    if 'KPI' in labels and 'type' in properties:
                        try:
                            session.run(f"DROP CONSTRAINT {constraint_name}")
                            print(f"Dropped incorrect constraint: {constraint_name} (KPI.type)")
                        except Exception as e:
                            print(f"Could not drop {constraint_name}: {e}")
            except Exception as e:
                print(f"Error checking constraints: {e}")
        
        # Create correct constraints
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (i:Industry) REQUIRE i.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Snippet) REQUIRE s.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (k:KPI) REQUIRE k.id IS UNIQUE",
            # Indexes for performance
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        with self.driver.session() as session:
            for q in queries:
                try:
                    session.run(q)
                except Exception as e:
                    print(f"Warning creating constraint: {e}")
        print("Constraints and indexes created.")

    def load_data(self):
        """Load processed data from parquet files: KPI entities, snippets, and entity mapping."""
        
        print("Loading data...")
        
        # Load extraction results
        kpi_path = config.KPI_ENTITIES_FILE
        if not os.path.exists(kpi_path):
            print(f"No extracted data found at {kpi_path}")
            return None, None, None
            
        df_kpi = pd.read_parquet(kpi_path)
        
        # Load snippets for metadata
        snippets_path = config.SNIPPETS_FILE
        if not os.path.exists(snippets_path):
            print(f"No snippets data found at {snippets_path}")
            return None, None, None
            
        df_snippets = pd.read_parquet(snippets_path)
        
        # Load entity map
        map_path = config.ENTITY_MAP_FILE
        if os.path.exists(map_path):
            df_map = pd.read_parquet(map_path)
            # Create a dictionary for fast lookup
            entity_map = dict(zip(df_map['raw_name'], df_map['canonical_name']))
        else:
            entity_map = {}
            
        return df_kpi, df_snippets, entity_map

    def _parse_json_list(self, raw):
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return []
        if isinstance(raw, list):
            return [str(x).strip() for x in raw if str(x).strip()]
        s = str(raw).strip()
        if not s or s.lower() == "nan":
            return []
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
        try:
            v = literal_eval(s)
            if isinstance(v, list):
                return [str(x).strip() for x in v if str(x).strip()]
        except Exception:
            pass
        return []

    def _safe_float(self, raw) -> float:
        if raw is None:
            return 0.0
        if isinstance(raw, float) and np.isnan(raw):
            return 0.0
        try:
            return float(raw)
        except Exception:
            return 0.0

    def _safe_int(self, raw) -> int:
        if raw is None:
            return 0
        if isinstance(raw, float) and np.isnan(raw):
            return 0
        try:
            return int(raw)
        except Exception:
            return 0

    def _is_junk_entity_name(self, name: str) -> bool:
        n = str(name or "").strip()
        if not n:
            return True
        low = n.lower()
        junk = {
            "ai",
            "artificial intelligence",
            "generative ai",
            "ai systems",
            "businesses",
            "companies",
            "company",
            "the company",
            "your company",
            "tech companies",
            "data centers",
            "unknown",
        }
        if low in junk:
            return True
        if len(n) <= 2:
            return True
        if low.endswith(" company") or low.endswith(" companies"):
            return True
        return False

    def _entity_type_priority(self, raw_type: str) -> int:
        """
        Normalize and assign a priority to entity types.

        Higher priority wins when an entity appears with multiple types across snippets.
        This helps prevent common issues where something like "FBI" is first seen as
        "Organization" and never corrected to "GovernmentAgency".
        """
        t = str(raw_type or "").strip()
        if not t:
            return 0
        low = t.lower()

        # Strong exclusions (should override generic "Organization").
        if "person" in low or low == "name":
            return 90
        if "government" in low or "agency" in low:
            return 80
        if low in {"country"} or "country" in low:
            return 70
        if "location" in low:
            return 60
        if "event" in low:
            return 55

        # Products/software are usually not what we want to rank as "startups".
        if "product" in low or "software" in low or "website" in low or "language" in low:
            return 40

        # Positive business entities.
        if "company" in low:
            return 30
        if "startup" in low:
            return 25
        if "investor" in low:
            return 20
        if "organization" in low or low in {"org"}:
            return 10

        return 1

    def build_graph(self):
        """Build knowledge graph by creating nodes and edges from extracted data.
        Creates Article, Entity, Industry, Snippet, and KPI nodes with relationships.
        """
        
        df_kpi, df_snippets, entity_map = self.load_data()
        if df_kpi is None:
            return

        print(f"Building graph from {len(df_kpi)} extraction records and {len(df_snippets)} snippets...")

        if "article_id" not in df_snippets.columns:
            raise ValueError("snippets.parquet is missing 'article_id'. Run upgrade_legacy_data.py / preprocess.py first.")

        # Join extractions with snippet metadata so we can build proper Article vs Snippet nodes.
        meta_cols = [
            "snippet_id",
            "article_id",
            "source",
            "title",
            "link",
            "canonical_url",
            "published",
            "chunk_index",
            "text",
            "parent_fetched_at",
        ]
        meta_cols = [c for c in meta_cols if c in df_snippets.columns]
        df_join = df_kpi.merge(df_snippets[meta_cols], how="left", on="snippet_id")
        df_join = df_join[df_join["article_id"].notna()].copy()

        # Track entities per article for co-occurrence calculation
        article_entities: dict[str, set[str]] = {}
        article_dates: dict[str, datetime] = {}

        def make_kpi_id(snippet_id: str, kpi_type: str, kpi_value: str, entity_name: str | None) -> str:
            base = f"{snippet_id}|{kpi_type}|{kpi_value}|{entity_name or ''}"
            return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()

        with self.driver.session() as session:
            # Build Article and Snippet nodes first.
            articles = (
                df_snippets.drop_duplicates("article_id")
                .sort_values("article_id")
                .to_dict("records")
            )
            for art in articles:
                article_id = str(art.get("article_id"))
                if not article_id or article_id == "nan":
                    continue
                session.run(
                    """
                    MERGE (a:Article {id: $id})
                    SET a.title = $title,
                        a.source = $source,
                        a.link = $link,
                        a.canonical_url = $canonical_url,
                        a.published = $published,
                        a.fetched_at = $fetched_at
                    """,
                    {
                        "id": article_id,
                        "title": art.get("title", "Unknown Title"),
                        "source": art.get("source", "Unknown Source"),
                        "link": art.get("link", ""),
                        "canonical_url": art.get("canonical_url", ""),
                        "published": str(art.get("published", "")),
                        "fetched_at": str(art.get("parent_fetched_at", "")),
                    },
                )

            for _, sn in df_snippets.iterrows():
                snippet_id = str(sn.get("snippet_id"))
                article_id = str(sn.get("article_id"))
                if not snippet_id or snippet_id == "nan" or not article_id or article_id == "nan":
                    continue
                chunk_index_raw = sn.get("chunk_index", 0)
                try:
                    chunk_index = int(chunk_index_raw)
                except Exception:
                    chunk_index = 0
                session.run(
                    """
                    MATCH (a:Article {id: $article_id})
                    MERGE (sn:Snippet {id: $snippet_id})
                    SET sn.text = $text,
                        sn.date = $published,
                        sn.chunk_index = $chunk_index
                    MERGE (sn)-[:IN]->(a)
                    """,
                    {
                        "article_id": article_id,
                        "snippet_id": snippet_id,
                        "text": sn.get("text", ""),
                        "published": str(sn.get("published", "")),
                        "chunk_index": chunk_index,
                    },
                )

            # Group by article_id to attach industry + mentions at article level.
            grouped_articles = df_join.groupby("article_id")
            processed = 0
            for article_id, group in grouped_articles:
                processed += 1
                if processed % 25 == 0:
                    print(f"Processed {processed} articles...")

                article_id = str(article_id)

                # Industry per article: prefer non-empty, non-Unknown values
                industry_vals = (
                    group.get("industry")
                    .dropna()
                    .astype(str)
                    .map(str.strip)
                    .loc[lambda s: (s != "") & (s.str.lower() != "unknown") & (s.str.lower() != "unclassified")]
                )
                main_industry = industry_vals.mode().iloc[0] if not industry_vals.empty else "Unclassified"

                session.run(
                    """
                    MATCH (a:Article {id: $article_id})
                    MERGE (i:Industry {name: $industry})
                    MERGE (a)-[:BELONGS_TO]->(i)
                    """,
                    {"article_id": article_id, "industry": main_industry},
                )

                # Aggregate SWOT per article (stored on Article for UI context).
                swot_data = {"Strength": [], "Weakness": [], "Opportunity": [], "Threat": []}
                swot_rows = group[group["category"] == "SWOT"]
                for _, row in swot_rows.iterrows():
                    dtype = str(row.get("detail_type", "")).strip()
                    dval = str(row.get("detail_value", "")).strip()
                    if dtype in swot_data and dval:
                        swot_data[dtype].append(dval)

                if any(swot_data.values()):
                    # Deduplicate while preserving order
                    for k in swot_data:
                        seen = set()
                        deduped = []
                        for v in swot_data[k]:
                            if v not in seen:
                                seen.add(v)
                                deduped.append(v)
                        swot_data[k] = deduped

                    session.run(
                        """
                        MATCH (a:Article {id: $article_id})
                        SET a.swot_Strength = $swot_Strength,
                            a.swot_Weakness = $swot_Weakness,
                            a.swot_Opportunity = $swot_Opportunity,
                            a.swot_Threat = $swot_Threat
                        """,
                        {
                            "article_id": article_id,
                            "swot_Strength": swot_data["Strength"],
                            "swot_Weakness": swot_data["Weakness"],
                            "swot_Opportunity": swot_data["Opportunity"],
                            "swot_Threat": swot_data["Threat"],
                        },
                    )

                # Article published date for recency decay in CO_LINK.
                pub = str(group.get("published").dropna().astype(str).head(1).iloc[0]) if "published" in group.columns and not group.get("published").dropna().empty else ""
                try:
                    article_dates[article_id] = datetime.fromisoformat(pub.replace("Z", "+00:00")) if pub else datetime.now()
                except Exception:
                    article_dates[article_id] = datetime.now()

                # Entity mentions
                entity_rows = group[group["category"] == "Entity"].copy()
                entity_rows = entity_rows[entity_rows["entity_name"].notna()]
                if not entity_rows.empty:
                    # Normalize via entity_map and aggregate stance.
                    entity_rows["entity_name"] = entity_rows["entity_name"].astype(str).map(lambda s: entity_map.get(s, s))
                    ent_agg = (
                        entity_rows.groupby(["entity_name"], as_index=False)
                        .agg(entity_type=("entity_type", "first"), stance=("stance", "mean"))
                    )
                    entities_to_batch = []
                    entity_names_set: set[str] = set()
                    for _, er in ent_agg.iterrows():
                        name = str(er.get("entity_name", "")).strip()
                        if not name:
                            continue
                        if self._is_junk_entity_name(name):
                            continue
                        etype = str(er.get("entity_type", "") or "Unknown")
                        entities_to_batch.append(
                            {
                                "name": name,
                                "type": etype,
                                "type_pri": int(self._entity_type_priority(etype)),
                                "stance": float(er.get("stance", 0.0) or 0.0),
                            }
                        )
                        entity_names_set.add(name)
                    article_entities[article_id] = entity_names_set

                    if entities_to_batch:
                        session.run(
                            """
                            MATCH (a:Article {id: $article_id})
                            UNWIND $entities AS ent
                            MERGE (e:Entity {name: ent.name})
                            ON CREATE SET e.type = ent.type, e.type_priority = ent.type_pri
                            ON MATCH SET
                                e.type = CASE WHEN coalesce(e.type_priority, 0) < ent.type_pri THEN ent.type ELSE e.type END,
                                e.type_priority = CASE WHEN coalesce(e.type_priority, 0) < ent.type_pri THEN ent.type_pri ELSE e.type_priority END
                            MERGE (a)-[r:MENTIONS]->(e)
                            SET r.stance = ent.stance
                            """,
                            {"article_id": article_id, "entities": entities_to_batch},
                        )

                # KPI nodes: connect Snippet -> KPI and (when available) Entity -> KPI.
                kpi_rows = group[group["category"] == "KPI"].copy()
                for _, row in kpi_rows.iterrows():
                    snippet_id = str(row.get("snippet_id", ""))
                    if not snippet_id or snippet_id == "nan":
                        continue
                    kpi_type = str(row.get("detail_type", "")).strip()
                    kpi_value = str(row.get("detail_value", "")).strip()
                    if not kpi_type or not kpi_value:
                        continue

                    entity_name = row.get("entity_name")
                    entity_name = str(entity_name).strip() if isinstance(entity_name, str) and entity_name.strip() else None
                    if entity_name:
                        entity_name = entity_map.get(entity_name, entity_name)

                    kpi_id = make_kpi_id(snippet_id, kpi_type, kpi_value, entity_name)
                    stance_val = float(row.get("stance", 0.0) or 0.0)
                    conf_val = float(row.get("confidence", 1.0) or 1.0)

                    session.run(
                        """
                        MATCH (sn:Snippet {id: $snippet_id})-[:IN]->(a:Article)
                        MERGE (k:KPI:Event {id: $kpi_id})
                        SET k.type = $kpi_type,
                            k.value = $kpi_value,
                            k.stance = $stance,
                            k.confidence = $confidence,
                            k.date = sn.date,
                            k.source_url = a.link,
                            k.article_id = a.id,
                            k.snippet_id = sn.id,
                            k.amount = $amount,
                            k.stage = $stage,
                            k.investors = $investors,
                            k.count = $count,
                            k.roles = $roles,
                            k.skills = $skills,
                            k.partner = $partner,
                            k.target = $target,
                            k.competitor = $competitor,
                            k.product_name = $product_name,
                            k.description = $description
                        MERGE (sn)-[:ABOUT]->(k)
                        """,
                        {
                            "snippet_id": snippet_id,
                            "kpi_id": kpi_id,
                            "kpi_type": kpi_type,
                            "kpi_value": kpi_value,
                            "stance": stance_val,
                            "confidence": conf_val,
                            "amount": self._safe_float(row.get("kpi_amount")) if "kpi_amount" in row else 0.0,
                            "stage": str(row.get("kpi_stage", "") or "").strip() if "kpi_stage" in row else "",
                            "investors": self._parse_json_list(row.get("kpi_investors")) if "kpi_investors" in row else [],
                            "count": self._safe_int(row.get("kpi_count")) if "kpi_count" in row else 0,
                            "roles": self._parse_json_list(row.get("kpi_roles")) if "kpi_roles" in row else [],
                            "skills": self._parse_json_list(row.get("kpi_skills")) if "kpi_skills" in row else [],
                            "partner": str(row.get("kpi_partner", "") or "").strip() if "kpi_partner" in row else "",
                            "target": str(row.get("kpi_target", "") or "").strip() if "kpi_target" in row else "",
                            "competitor": str(row.get("kpi_competitor", "") or "").strip() if "kpi_competitor" in row else "",
                            "product_name": str(row.get("kpi_product_name", "") or "").strip() if "kpi_product_name" in row else "",
                            "description": str(row.get("kpi_description", "") or "").strip() if "kpi_description" in row else "",
                        },
                    )

                    if entity_name:
                        session.run(
                            """
                            MATCH (e:Entity {name: $entity_name})
                            MATCH (k:KPI {id: $kpi_id})
                            MERGE (e)-[:HAS_KPI]->(k)
                            """,
                            {"entity_name": entity_name, "kpi_id": kpi_id},
                        )

                    # Role-aware counterpart links (typed, per-event) to reduce "graph spaghetti".
                    # We attach the counterpart to the Event/KPI node, not via co-mention.
                    if kpi_type == "Funding":
                        investors = self._parse_json_list(row.get("kpi_investors"))
                        if investors:
                            inv_payload = [{"name": n} for n in investors if n and not self._is_junk_entity_name(n)]
                            if inv_payload:
                                session.run(
                                    """
                                    MATCH (k:KPI {id: $kpi_id})
                                    UNWIND $investors AS inv
                                    MERGE (i:Investor {name: inv.name})
                                    MERGE (k)-[:COUNTERPART {role: 'Investor'}]->(i)
                                    """,
                                    {"kpi_id": kpi_id, "investors": inv_payload},
                                )

                    if kpi_type == "Partnership":
                        partner = str(row.get("kpi_partner", "") or "").strip()
                        if partner and not self._is_junk_entity_name(partner):
                            session.run(
                                """
                                MATCH (k:KPI {id: $kpi_id})
                                MERGE (p:Entity {name: $partner})
                                MERGE (k)-[:COUNTERPART {role: 'Partner'}]->(p)
                                """,
                                {"kpi_id": kpi_id, "partner": partner},
                            )

                    if kpi_type == "Acquisition":
                        target = str(row.get("kpi_target", "") or "").strip()
                        if target and not self._is_junk_entity_name(target):
                            session.run(
                                """
                                MATCH (k:KPI {id: $kpi_id})
                                MERGE (t:Entity {name: $target})
                                MERGE (k)-[:COUNTERPART {role: 'Target'}]->(t)
                                """,
                                {"kpi_id": kpi_id, "target": target},
                            )

                    if kpi_type == "Competition":
                        competitor = str(row.get("kpi_competitor", "") or "").strip()
                        if competitor and not self._is_junk_entity_name(competitor):
                            session.run(
                                """
                                MATCH (k:KPI {id: $kpi_id})
                                MERGE (c:Entity {name: $competitor})
                                MERGE (k)-[:COUNTERPART {role: 'Competitor'}]->(c)
                                """,
                                {"kpi_id": kpi_id, "competitor": competitor},
                            )

        print("Graph build phase 1 complete. Now creating Article-Article edges...")
        self._create_article_colinks(article_entities, article_dates)
        print("Graph build complete.")
    
    def _determine_kpi_stance(self, kpi_type: str, kpi_value: str) -> float:
        """Determine positive/negative/neutral stance from KPI context."""
        kpi_value_lower = kpi_value.lower() if kpi_value else ""
        
        # Positive indicators
        positive_words = ['raised', 'secured', 'growth', 'increase', 'expansion', 'launched', 
                         'acquired', 'partnership', 'funding', 'investment', 'hired']
        # Negative indicators  
        negative_words = ['layoff', 'shutdown', 'decline', 'loss', 'lawsuit', 'breach', 
                         'failure', 'delay', 'issue', 'problem', 'risk']
        
        pos_count = sum(1 for word in positive_words if word in kpi_value_lower)
        neg_count = sum(1 for word in negative_words if word in kpi_value_lower)
        
        if pos_count > neg_count:
            return 1.0
        elif neg_count > pos_count:
            return -1.0
        else:
            return 0.0
    
    def _create_article_colinks(self, article_entities: dict, article_dates: dict):
        """Create weighted Article-Article edges based on Jaccard similarity of shared entities.

        Uses a per-article top-k strategy (instead of a global weight threshold) to avoid
        overly sparse graphs that fragment communities on small datasets.
        """
        print("Calculating Jaccard similarity for article pairs...")
        
        article_ids = list(article_entities.keys())
        neighbors: dict[str, list[dict]] = {aid: [] for aid in article_ids}
        tau = 30  # Recency decay parameter (days)
        now = pd.Timestamp.now(tz='UTC')  # Use timezone-aware timestamp
        
        for i, aid1 in enumerate(article_ids):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(article_ids)} articles...")
            
            entities1 = article_entities[aid1]
            if len(entities1) == 0:
                continue
            
            date1 = article_dates.get(aid1, now)
            # Ensure date1 is timezone-aware
            if date1.tzinfo is None:
                date1 = pd.Timestamp(date1, tz='UTC')
            
            # Only compare with subsequent articles to avoid duplicates
            for aid2 in article_ids[i+1:]:
                entities2 = article_entities[aid2]
                if len(entities2) == 0:
                    continue
                
                # Calculate Jaccard similarity
                intersection = len(entities1 & entities2)
                if intersection == 0:
                    continue
                    
                union = len(entities1 | entities2)
                jaccard = intersection / union
                
                # Apply recency decay
                date2 = article_dates.get(aid2, now)
                # Ensure date2 is timezone-aware
                if date2.tzinfo is None:
                    date2 = pd.Timestamp(date2, tz='UTC')
                    
                avg_date = date1 if date1 > date2 else date2  # Use newer date
                days_old = (now - avg_date).days
                recency_weight = np.exp(-days_old / tau)
                
                # Final weight
                weight = jaccard * recency_weight

                if weight <= 0:
                    continue

                neighbors[aid1].append({'aid': aid2, 'weight': float(weight), 'shared_entities': int(intersection)})
                neighbors[aid2].append({'aid': aid1, 'weight': float(weight), 'shared_entities': int(intersection)})
        
        # Keep top-k neighbors per article to ensure connectivity without exploding edges.
        top_k = 5
        edge_map: dict[tuple[str, str], dict] = {}
        for aid, nbrs in neighbors.items():
            nbrs.sort(key=lambda x: x['weight'], reverse=True)
            for n in nbrs[:top_k]:
                a1, a2 = (aid, n['aid']) if aid < n['aid'] else (n['aid'], aid)
                key = (a1, a2)
                if key not in edge_map or n['weight'] > edge_map[key]['weight']:
                    edge_map[key] = {'aid1': a1, 'aid2': a2, 'weight': n['weight'], 'shared_entities': n['shared_entities']}

        edges_to_create = list(edge_map.values())

        print(f"Creating {len(edges_to_create)} CO_LINK relationships...")
        
        with self.driver.session() as session:
            # Batch create edges
            for i in range(0, len(edges_to_create), 500):
                batch = edges_to_create[i:i+500]
                session.run("""
                    UNWIND $edges AS edge
                    MATCH (a1:Article {id: edge.aid1})
                    MATCH (a2:Article {id: edge.aid2})
                    MERGE (a1)-[r:CO_LINK]-(a2)
                    SET r.weight = edge.weight, r.shared_entities = edge.shared_entities
                """, {'edges': batch})
                if i % 1000 == 0 and i > 0:
                    print(f"  Created {i} edges...")
        
        print(f"[OK] Created {len(edges_to_create)} weighted Article-Article edges")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build TrendScout knowledge graph in Neo4j.")
    parser.add_argument("--reset", action="store_true", help="Delete ALL nodes/relationships in Neo4j before building.")
    parser.add_argument("--yes", action="store_true", help="Required confirmation flag when using --reset.")
    args = parser.parse_args()

    builder = GraphBuilder()
    try:
        if args.reset:
            if not args.yes:
                raise SystemExit("Refusing to reset Neo4j without --yes. Example: python CODE/graph_build.py --reset --yes")
            print("[WARN] Resetting Neo4j database (MATCH (n) DETACH DELETE n)...")
            builder.clear_database()

        builder.create_constraints()
        builder.build_graph()
    finally:
        builder.close()
