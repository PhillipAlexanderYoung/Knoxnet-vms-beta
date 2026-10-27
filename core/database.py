import sqlite3
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid

logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, db_path: str = 'data/sentry.db'):
        self.db_path = db_path
        self.lock = threading.Lock()

    def initialize(self):
        """Initialize database tables"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Create tables
                self.create_tables(cursor)

                conn.commit()
                conn.close()

            logger.info("Database initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def create_tables(self, cursor):
        """Create database tables"""
        # Alerts table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS alerts
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           type
                           TEXT
                           NOT
                           NULL,
                           camera_id
                           TEXT
                           NOT
                           NULL,
                           timestamp
                           TEXT
                           NOT
                           NULL,
                           priority
                           TEXT
                           NOT
                           NULL,
                           description
                           TEXT
                           NOT
                           NULL,
                           data
                           TEXT,
                           acknowledged
                           BOOLEAN
                           DEFAULT
                           FALSE,
                           acknowledged_at
                           TEXT,
                           resolved
                           BOOLEAN
                           DEFAULT
                           FALSE,
                           resolved_at
                           TEXT
                       )
                       ''')

        # Detection events table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS detection_events
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           camera_id
                           TEXT
                           NOT
                           NULL,
                           timestamp
                           TEXT
                           NOT
                           NULL,
                           detections
                           TEXT
                           NOT
                           NULL,
                           frame_path
                           TEXT,
                           analysis_duration
                           REAL
                       )
                       ''')

        # Camera configurations table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS camera_configs
                       (
                           camera_id
                           TEXT
                           PRIMARY
                           KEY,
                           name
                           TEXT
                           NOT
                           NULL,
                           config
                           TEXT
                           NOT
                           NULL,
                           created_at
                           TEXT
                           NOT
                           NULL,
                           updated_at
                           TEXT
                           NOT
                           NULL
                       )
                       ''')

        # Camera shapes table (zones/lines/tags)
        # Stores UI-drawn shapes in a server-side, camera-scoped record so server-side automations
        # can reliably evaluate zone/line/tag interactions even when the UI is closed.
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS camera_shapes (
                           camera_id TEXT PRIMARY KEY,
                           shapes_json TEXT NOT NULL,
                           created_at TEXT NOT NULL,
                           updated_at TEXT NOT NULL
                       )
                       ''')

        # System events table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS system_events
                       (
                           id
                           TEXT
                           PRIMARY
                           KEY,
                           event_type
                           TEXT
                           NOT
                           NULL,
                           timestamp
                           TEXT
                           NOT
                           NULL,
                           description
                           TEXT
                           NOT
                           NULL,
                           data
                           TEXT
                       )
                       ''')

        # User sessions table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS user_sessions
                       (
                           session_id
                           TEXT
                           PRIMARY
                           KEY,
                           user_id
                           TEXT,
                           created_at
                           TEXT
                           NOT
                           NULL,
                           last_activity
                           TEXT
                           NOT
                           NULL,
                           ip_address
                           TEXT,
                           user_agent
                           TEXT
                       )
                       ''')

        # Rules table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS rules (
                           id TEXT PRIMARY KEY,
                           name TEXT NOT NULL,
                           camera_id TEXT,
                           shape_id TEXT,
                           trigger TEXT NOT NULL,
                           conditions TEXT,
                           actions TEXT,
                           enabled BOOLEAN DEFAULT TRUE,
                           created_at TEXT NOT NULL,
                           updated_at TEXT NOT NULL
                       )
                       ''')

        # Event bundles table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS event_bundles (
                           id TEXT PRIMARY KEY,
                           camera_id TEXT NOT NULL,
                           kind TEXT NOT NULL,
                           created_at TEXT NOT NULL,
                           bundle_json TEXT NOT NULL
                       )
                       ''')

        # Analysis memories table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS analysis_memories (
                           id TEXT PRIMARY KEY,
                           bundle_id TEXT,
                           camera_id TEXT,
                           created_at TEXT NOT NULL,
                           message TEXT,
                           vision_analysis TEXT,
                           verdict_json TEXT
                       )
                       ''')

        # Zone violations table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS zone_violations (
                           id TEXT PRIMARY KEY,
                           camera_id TEXT NOT NULL,
                           zone_id TEXT,
                           zone_name TEXT,
                           violation_type TEXT,
                           confidence REAL,
                           description TEXT,
                           timestamp TEXT NOT NULL,
                           bundle_id TEXT
                       )
                       ''')

        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_camera ON alerts(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_events_timestamp ON detection_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_detection_events_camera ON detection_events(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_system_events_timestamp ON system_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rules_camera ON rules(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rules_shape ON rules(shape_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_bundles_camera ON event_bundles(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_event_bundles_kind ON event_bundles(kind)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_analysis_memories_camera ON analysis_memories(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_zone_violations_camera ON zone_violations(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_zone_violations_zone ON zone_violations(zone_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_camera_shapes_updated ON camera_shapes(updated_at)')

        # Server-side screenshots table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS server_screenshots (
                           id TEXT PRIMARY KEY,
                           created_at TEXT NOT NULL,
                           camera_id TEXT,
                           camera_name TEXT,
                           file_path TEXT NOT NULL,
                           width INTEGER,
                           height INTEGER,
                           size_bytes INTEGER,
                           format TEXT,
                           motion_type TEXT,
                           motion_confidence REAL,
                           motion_size REAL,
                           scene_description TEXT,
                           local_vision_analysis TEXT,
                           api_vision_analysis TEXT,
                           analysis_keywords TEXT,
                           metadata_json TEXT
                       )
                       ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_srvshots_created ON server_screenshots(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_srvshots_camera ON server_screenshots(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_srvshots_motion ON server_screenshots(motion_type)')
        
        # Migrate existing table: add new columns if they don't exist
        try:
            cursor.execute("SELECT local_vision_analysis FROM server_screenshots LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            logger.info("Migrating server_screenshots table: adding local_vision_analysis column")
            cursor.execute("ALTER TABLE server_screenshots ADD COLUMN local_vision_analysis TEXT")
        
        try:
            cursor.execute("SELECT api_vision_analysis FROM server_screenshots LIMIT 1")
        except sqlite3.OperationalError:
            # Column doesn't exist, add it
            logger.info("Migrating server_screenshots table: adding api_vision_analysis column")
            cursor.execute("ALTER TABLE server_screenshots ADD COLUMN api_vision_analysis TEXT")

        # Detection feedback table (user corrections and hints)
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS detection_feedback (
                           id TEXT PRIMARY KEY,
                           camera_id TEXT NOT NULL,
                           timestamp TEXT NOT NULL,
                           kind TEXT NOT NULL,                 -- 'correction' | 'hint'
                           object_class TEXT,                  -- normalized class name (e.g., 'person','car')
                           correct BOOLEAN,                    -- for corrections: whether the label was correct
                           bbox TEXT,                          -- JSON: {x,y,w,h} normalized (0..1)
                           detection_meta TEXT,                -- JSON: detector metadata {model, track_id, confidence}
                           note TEXT                           -- optional free text
                       )
                       ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_camera ON detection_feedback(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_kind ON detection_feedback(kind)')

        # Track trajectories table
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS track_trajectories (
                           id TEXT PRIMARY KEY,
                           camera_id TEXT NOT NULL,
                           track_id INTEGER NOT NULL,
                           short_id TEXT,
                           started_at TEXT,
                           ended_at TEXT,
                           active BOOLEAN,
                           points TEXT,
                           last_updated TEXT
                       )
                       ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_traj_camera ON track_trajectories(camera_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_track_traj_trackid ON track_trajectories(track_id)')

    # ==================== RULES CRUD ====================

    def create_rule(self, rule: Dict[str, Any]) -> Optional[str]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                rule_id = rule.get('id') or str(uuid.uuid4())
                now = datetime.now().isoformat()
                cursor.execute('''
                    INSERT INTO rules (id, name, camera_id, shape_id, trigger, conditions, actions, enabled, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rule_id,
                    rule.get('name', 'Rule'),
                    rule.get('camera_id'),
                    rule.get('shape_id'),
                    rule.get('trigger', 'motion_detected'),
                    json.dumps(rule.get('conditions', {})),
                    json.dumps(rule.get('actions', [])),
                    bool(rule.get('enabled', True)),
                    now,
                    now
                ))
                conn.commit()
                conn.close()
                return rule_id
        except Exception as e:
            logger.error(f"Error creating rule: {e}")
            return None

    def get_rule(self, rule_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM rules WHERE id = ?', (rule_id,))
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return None
                columns = [d[0] for d in cursor.description]
                rule = dict(zip(columns, row))
                rule['conditions'] = json.loads(rule['conditions'] or '{}')
                rule['actions'] = json.loads(rule['actions'] or '[]')
                conn.close()
                return rule
        except Exception as e:
            logger.error(f"Error getting rule {rule_id}: {e}")
            return None

    def list_rules(self, camera_id: str = None, shape_id: str = None, enabled: bool = None) -> List[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                query = 'SELECT * FROM rules WHERE 1=1'
                params: List[Any] = []
                if camera_id:
                    query += ' AND camera_id = ?'
                    params.append(camera_id)
                if shape_id:
                    query += ' AND shape_id = ?'
                    params.append(shape_id)
                if enabled is not None:
                    query += ' AND enabled = ?'
                    params.append(1 if enabled else 0)
                query += ' ORDER BY updated_at DESC'
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                rules: List[Dict[str, Any]] = []
                for row in rows:
                    r = dict(zip(columns, row))
                    r['conditions'] = json.loads(r['conditions'] or '{}')
                    r['actions'] = json.loads(r['actions'] or '[]')
                    rules.append(r)
                conn.close()
                return rules
        except Exception as e:
            logger.error(f"Error listing rules: {e}")
            return []

    def update_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                fields = []
                params: List[Any] = []
                for key in ['name', 'camera_id', 'shape_id', 'trigger', 'enabled']:
                    if key in updates:
                        fields.append(f"{key} = ?")
                        params.append(updates[key])
                if 'conditions' in updates:
                    fields.append('conditions = ?')
                    params.append(json.dumps(updates['conditions']))
                if 'actions' in updates:
                    fields.append('actions = ?')
                    params.append(json.dumps(updates['actions']))
                fields.append('updated_at = ?')
                params.append(datetime.now().isoformat())
                if not fields:
                    conn.close()
                    return False
                params.append(rule_id)
                cursor.execute(f"UPDATE rules SET {', '.join(fields)} WHERE id = ?", params)
                conn.commit()
                success = cursor.rowcount > 0
                conn.close()
                return success
        except Exception as e:
            logger.error(f"Error updating rule {rule_id}: {e}")
            return False

    def delete_rule(self, rule_id: str) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM rules WHERE id = ?', (rule_id,))
                conn.commit()
                success = cursor.rowcount > 0
                conn.close()
                return success
        except Exception as e:
            logger.error(f"Error deleting rule {rule_id}: {e}")
            return False

    # ==================== CAMERA SHAPES (ZONES/LINES/TAGS) ====================

    def upsert_camera_shapes(self, camera_id: str, shapes: Dict[str, Any]) -> bool:
        """
        Store per-camera UI shapes (zones/lines/tags). Shapes should be JSON-serializable.
        Expected top-level keys: zones, lines, tags (arrays).
        """
        try:
            cam_id = str(camera_id)
            now = datetime.now().isoformat()
            payload = {
                "camera_id": cam_id,
                "zones": shapes.get("zones", []) if isinstance(shapes, dict) else [],
                "lines": shapes.get("lines", []) if isinstance(shapes, dict) else [],
                "tags": shapes.get("tags", []) if isinstance(shapes, dict) else [],
            }
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    INSERT INTO camera_shapes (camera_id, shapes_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(camera_id) DO UPDATE SET
                        shapes_json = excluded.shapes_json,
                        updated_at = excluded.updated_at
                    ''',
                    (cam_id, json.dumps(payload), now, now),
                )
                conn.commit()
                conn.close()
            return True
        except Exception as e:
            logger.error(f"Error upserting camera shapes for {camera_id}: {e}")
            return False

    def get_camera_shapes(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """Fetch stored camera shapes (zones/lines/tags). Returns None if not found."""
        try:
            cam_id = str(camera_id)
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT shapes_json FROM camera_shapes WHERE camera_id = ?', (cam_id,))
                row = cursor.fetchone()
                conn.close()
            if not row:
                return None
            raw = row[0]
            parsed = json.loads(raw or "{}")
            if not isinstance(parsed, dict):
                return None
            parsed.setdefault("camera_id", cam_id)
            parsed.setdefault("zones", [])
            parsed.setdefault("lines", [])
            parsed.setdefault("tags", [])
            return parsed
        except Exception as e:
            logger.error(f"Error getting camera shapes for {camera_id}: {e}")
            return None

    def delete_camera_shapes(self, camera_id: str) -> bool:
        try:
            cam_id = str(camera_id)
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM camera_shapes WHERE camera_id = ?', (cam_id,))
                conn.commit()
                success = cursor.rowcount > 0
                conn.close()
                return success
        except Exception as e:
            logger.error(f"Error deleting camera shapes for {camera_id}: {e}")
            return False

    # ==================== EVENT BUNDLES ====================

    def store_event_bundle(self, bundle_id: str, camera_id: str, kind: str, created_at: str, bundle_json: str) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO event_bundles (id, camera_id, kind, created_at, bundle_json)
                    VALUES (?, ?, ?, ?, ?)
                ''', (bundle_id, camera_id, kind, created_at, bundle_json))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Error storing event bundle {bundle_id}: {e}")
            return False

    def get_event_bundle(self, bundle_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM event_bundles WHERE id = ?', (bundle_id,))
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return None
                columns = [d[0] for d in cursor.description]
                rec = dict(zip(columns, row))
                # Parse JSON
                if rec.get('bundle_json'):
                    rec['bundle'] = json.loads(rec['bundle_json'])
                conn.close()
                return rec
        except Exception as e:
            logger.error(f"Error getting event bundle {bundle_id}: {e}")
            return None

    def list_event_bundles(self, camera_id: str = None, kind: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                query = 'SELECT id, camera_id, kind, created_at FROM event_bundles WHERE 1=1'
                params: List[Any] = []
                if camera_id:
                    query += ' AND camera_id = ?'
                    params.append(camera_id)
                if kind:
                    query += ' AND kind = ?'
                    params.append(kind)
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                recs = [dict(zip(columns, row)) for row in rows]
                conn.close()
                return recs
        except Exception as e:
            logger.error(f"Error listing event bundles: {e}")
            return []

    def delete_event_bundle(self, bundle_id: str) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM event_bundles WHERE id = ?', (bundle_id,))
                conn.commit()
                success = cursor.rowcount > 0
                conn.close()
                return success
        except Exception as e:
            logger.error(f"Error deleting event bundle {bundle_id}: {e}")
            return False

    def update_event_bundle_json(self, bundle_id: str, bundle_json: Dict[str, Any]) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE event_bundles SET bundle_json = ? WHERE id = ?
                ''', (json.dumps(bundle_json), bundle_id))
                conn.commit()
                success = cursor.rowcount > 0
                conn.close()
                return success
        except Exception as e:
            logger.error(f"Error updating event bundle {bundle_id}: {e}")
            return False

    # ==================== TRACK TRAJECTORIES ====================

    def store_track_trajectory(self, camera_id: str, traj: Dict[str, Any]) -> bool:
        """Insert or update a track trajectory record.
        traj should include: id (uuid), track_id (int), short_id (str), started_at, ended_at, active (bool), points (list[dict])
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                rec_id = traj.get('id') or str(uuid.uuid4())
                cursor.execute('''
                    INSERT OR REPLACE INTO track_trajectories
                    (id, camera_id, track_id, short_id, started_at, ended_at, active, points, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rec_id,
                    camera_id,
                    int(traj.get('track_id') or traj.get('id') or 0),
                    traj.get('short_id'),
                    traj.get('started_at'),
                    traj.get('ended_at'),
                    bool(traj.get('active', False)),
                    json.dumps(traj.get('points', [])),
                    datetime.now().isoformat()
                ))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Error storing track trajectory for camera {camera_id}: {e}")
            return False

    def list_track_trajectories(self, camera_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT id, camera_id, track_id, short_id, started_at, ended_at, active, points, last_updated
                    FROM track_trajectories
                    WHERE camera_id = ?
                    ORDER BY last_updated DESC
                    LIMIT ?
                ''', (camera_id, limit))
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                recs = []
                for row in rows:
                    r = dict(zip(columns, row))
                    try:
                        if r.get('points'):
                            r['points'] = json.loads(r['points'])
                    except Exception:
                        pass
                    recs.append(r)
                conn.close()
                return recs
        except Exception as e:
            logger.error(f"Error listing track trajectories for camera {camera_id}: {e}")
            return []

    # ==================== ANALYSIS MEMORIES ====================

    def store_analysis_memory(self, memory_id: str, bundle_id: str, camera_id: str,
                               created_at: str, message: str, vision_analysis: str,
                               verdict_json: Dict[str, Any]) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO analysis_memories
                    (id, bundle_id, camera_id, created_at, message, vision_analysis, verdict_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (memory_id, bundle_id, camera_id, created_at, message, vision_analysis, json.dumps(verdict_json)))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Error storing analysis memory {memory_id}: {e}")
            return False

    def list_analysis_memories(self, camera_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                query = 'SELECT id, bundle_id, camera_id, created_at, message, vision_analysis, verdict_json FROM analysis_memories WHERE 1=1'
                params: List[Any] = []
                if camera_id:
                    query += ' AND camera_id = ?'
                    params.append(camera_id)
                query += ' ORDER BY created_at DESC LIMIT ?'
                params.append(limit)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                recs = []
                for row in rows:
                    r = dict(zip(columns, row))
                    if r.get('verdict_json'):
                        r['verdict'] = json.loads(r['verdict_json'])
                    recs.append(r)
                conn.close()
                return recs
        except Exception as e:
            logger.error(f"Error listing analysis memories: {e}")
            return []

    # ==================== ZONE VIOLATIONS ====================

    def store_zone_violation(self, violation: Dict[str, Any]) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO zone_violations
                    (id, camera_id, zone_id, zone_name, violation_type, confidence, description, timestamp, bundle_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    violation['id'],
                    violation['camera_id'],
                    violation.get('zone_id'),
                    violation.get('zone_name'),
                    violation.get('violation_type'),
                    violation.get('confidence', 0.0),
                    violation.get('description'),
                    violation.get('timestamp'),
                    violation.get('bundle_id')
                ))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Error storing zone violation: {e}")
            return False

    def list_zone_violations(self, camera_id: str = None, zone_id: str = None,
                              limit: int = 100) -> List[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                query = 'SELECT * FROM zone_violations WHERE 1=1'
                params: List[Any] = []
                if camera_id:
                    query += ' AND camera_id = ?'
                    params.append(camera_id)
                if zone_id:
                    query += ' AND zone_id = ?'
                    params.append(zone_id)
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                recs = [dict(zip(columns, row)) for row in rows]
                conn.close()
                return recs
        except Exception as e:
            logger.error(f"Error listing zone violations: {e}")
            return []

    # ==================== DETECTION FEEDBACK ====================
    def store_detection_feedback(self, feedback: Dict[str, Any]) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                rec_id = feedback.get('id') or str(uuid.uuid4())
                cursor.execute('''
                    INSERT OR REPLACE INTO detection_feedback
                    (id, camera_id, timestamp, kind, object_class, correct, bbox, detection_meta, note)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    rec_id,
                    feedback.get('camera_id'),
                    feedback.get('timestamp'),
                    feedback.get('kind'),
                    feedback.get('object_class'),
                    bool(feedback.get('correct')) if feedback.get('kind') == 'correction' else None,
                    json.dumps(feedback.get('bbox') or {}),
                    json.dumps(feedback.get('detection_meta') or {}),
                    feedback.get('note')
                ))
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            logger.error(f"Error storing detection feedback: {e}")
            return False

    def list_detection_feedback(self, camera_id: Optional[str] = None, kind: Optional[str] = None, limit: int = 200) -> List[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                query = 'SELECT * FROM detection_feedback WHERE 1=1'
                params: List[Any] = []
                if camera_id:
                    query += ' AND camera_id = ?'
                    params.append(camera_id)
                if kind:
                    query += ' AND kind = ?'
                    params.append(kind)
                query += ' ORDER BY timestamp DESC LIMIT ?'
                params.append(limit)
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                recs: List[Dict[str, Any]] = []
                for row in rows:
                    r = dict(zip(columns, row))
                    try:
                        if r.get('bbox'):
                            r['bbox'] = json.loads(r['bbox'])
                        if r.get('detection_meta'):
                            r['detection_meta'] = json.loads(r['detection_meta'])
                    except Exception:
                        pass
                    recs.append(r)
                conn.close()
                return recs
        except Exception as e:
            logger.error(f"Error listing detection feedback: {e}")
            return []

    def store_alert(self, alert: Dict[str, Any]) -> bool:
        """Store alert in database"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cursor.execute('''
                               INSERT INTO alerts
                               (id, type, camera_id, timestamp, priority, description, data, acknowledged, resolved)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                               ''', (
                                   alert['id'],
                                   alert['type'],
                                   alert['camera_id'],
                                   alert['timestamp'],
                                   alert['priority'],
                                   alert['description'],
                                   json.dumps(alert.get('data', {})),
                                   alert.get('acknowledged', False),
                                   alert.get('resolved', False)
                               ))

                conn.commit()
                conn.close()

            return True

        except Exception as e:
            logger.error(f"Error storing alert: {e}")
            return False

    def get_alerts(self, limit: int = 100, offset: int = 0,
                   camera_id: str = None, priority: str = None,
                   start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get alerts from database with filtering"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                query = "SELECT * FROM alerts WHERE 1=1"
                params = []

                if camera_id:
                    query += " AND camera_id = ?"
                    params.append(camera_id)

                if priority:
                    query += " AND priority = ?"
                    params.append(priority)

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)

                query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
                params.extend([limit, offset])

                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Convert to dictionaries
                columns = [description[0] for description in cursor.description]
                alerts = []

                for row in rows:
                    alert = dict(zip(columns, row))
                    if alert['data']:
                        alert['data'] = json.loads(alert['data'])
                    alerts.append(alert)

                conn.close()
                return alerts

        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    def update_alert(self, alert_id: str, updates: Dict[str, Any]) -> bool:
        """Update alert in database"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                # Build update query
                set_clauses = []
                params = []

                for key, value in updates.items():
                    if key in ['acknowledged', 'resolved', 'acknowledged_at', 'resolved_at']:
                        set_clauses.append(f"{key} = ?")
                        params.append(value)

                if not set_clauses:
                    return False

                query = f"UPDATE alerts SET {', '.join(set_clauses)} WHERE id = ?"
                params.append(alert_id)

                cursor.execute(query, params)
                conn.commit()

                success = cursor.rowcount > 0
                conn.close()

                return success

        except Exception as e:
            logger.error(f"Error updating alert: {e}")
            return False

    def store_detection_event(self, camera_id: str, detections: List[Dict[str, Any]],
                              analysis_duration: float = None) -> bool:
        """Store detection event in database"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                event_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()

                cursor.execute('''
                               INSERT INTO detection_events
                                   (id, camera_id, timestamp, detections, analysis_duration)
                               VALUES (?, ?, ?, ?, ?)
                               ''', (
                                   event_id,
                                   camera_id,
                                   timestamp,
                                   json.dumps(detections),
                                   analysis_duration
                               ))

                conn.commit()
                conn.close()

            return True

        except Exception as e:
            logger.error(f"Error storing detection event: {e}")
            return False

    def get_detection_events(self, camera_id: str = None, limit: int = 100,
                             start_date: str = None, end_date: str = None) -> List[Dict[str, Any]]:
        """Get detection events from database"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                query = "SELECT * FROM detection_events WHERE 1=1"
                params = []

                if camera_id:
                    query += " AND camera_id = ?"
                    params.append(camera_id)

                if start_date:
                    query += " AND timestamp >= ?"
                    params.append(start_date)

                if end_date:
                    query += " AND timestamp <= ?"
                    params.append(end_date)

                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Convert to dictionaries
                columns = [description[0] for description in cursor.description]
                events = []

                for row in rows:
                    event = dict(zip(columns, row))
                    if event['detections']:
                        event['detections'] = json.loads(event['detections'])
                    events.append(event)

                conn.close()
                return events

        except Exception as e:
            logger.error(f"Error getting detection events: {e}")
            return []

    def store_system_event(self, event_type: str, description: str, data: Dict[str, Any] = None) -> bool:
        """Store system event in database"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                event_id = str(uuid.uuid4())
                timestamp = datetime.now().isoformat()

                cursor.execute('''
                               INSERT INTO system_events (id, event_type, timestamp, description, data)
                               VALUES (?, ?, ?, ?, ?)
                               ''', (
                                   event_id,
                                   event_type,
                                   timestamp,
                                   description,
                                   json.dumps(data) if data else None
                               ))

                conn.commit()
                conn.close()

            return True

        except Exception as e:
            logger.error(f"Error storing system event: {e}")
            return False

    def get_statistics(self, camera_id: str = None, days: int = 7) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                start_date = (datetime.now() - timedelta(days=days)).isoformat()

                stats = {}

                # Alert statistics
                query = "SELECT priority, COUNT(*) FROM alerts WHERE timestamp >= ?"
                params = [start_date]

                if camera_id:
                    query += " AND camera_id = ?"
                    params.append(camera_id)

                query += " GROUP BY priority"

                cursor.execute(query, params)
                alert_stats = dict(cursor.fetchall())
                stats['alerts'] = alert_stats

                # Detection statistics
                query = "SELECT COUNT(*) FROM detection_events WHERE timestamp >= ?"
                params = [start_date]

                if camera_id:
                    query += " AND camera_id = ?"
                    params.append(camera_id)

                cursor.execute(query, params)
                detection_count = cursor.fetchone()[0]
                stats['total_detections'] = detection_count

                conn.close()
                return stats

        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

    def cleanup_old_data(self, days: int = 30) -> bool:
        """Clean up old data from database"""
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()

                cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

                # Clean up old detection events
                cursor.execute("DELETE FROM detection_events WHERE timestamp < ?", (cutoff_date,))
                detection_deleted = cursor.rowcount

                # Clean up resolved alerts older than cutoff
                cursor.execute(
                    "DELETE FROM alerts WHERE timestamp < ? AND resolved = TRUE",
                    (cutoff_date,)
                )
                alerts_deleted = cursor.rowcount

                # Clean up old system events
                cursor.execute("DELETE FROM system_events WHERE timestamp < ?", (cutoff_date,))
                events_deleted = cursor.rowcount

                conn.commit()
                conn.close()

                logger.info(f"Cleaned up old data: {detection_deleted} detections, "
                            f"{alerts_deleted} alerts, {events_deleted} system events")

                return True

        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return False

    # ==================== SERVER-SIDE SCREENSHOTS ====================

    def store_server_screenshot(self, rec: Dict[str, Any]) -> Optional[str]:
        """Insert a server-side screenshot record. Expects keys:
        id, created_at, camera_id, camera_name, file_path, width, height, size_bytes,
        format, motion_type, motion_confidence, motion_size, scene_description,
        local_vision_analysis, api_vision_analysis, analysis_keywords (list[str]), metadata_json (dict)
        """
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                rec_id = rec.get('id') or str(uuid.uuid4())
                cursor.execute('''
                    INSERT OR REPLACE INTO server_screenshots
                    (id, created_at, camera_id, camera_name, file_path, width, height, size_bytes, format,
                     motion_type, motion_confidence, motion_size, scene_description, local_vision_analysis, 
                     api_vision_analysis, analysis_keywords, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''' , (
                    rec_id,
                    rec.get('created_at') or datetime.now().isoformat(),
                    rec.get('camera_id'),
                    rec.get('camera_name'),
                    rec.get('file_path'),
                    int(rec.get('width') or 0),
                    int(rec.get('height') or 0),
                    int(rec.get('size_bytes') or 0),
                    rec.get('format') or 'jpeg',
                    rec.get('motion_type'),
                    float(rec.get('motion_confidence') or 0.0),
                    float(rec.get('motion_size') or 0.0),
                    rec.get('scene_description'),
                    rec.get('local_vision_analysis'),  # Save BLIP analysis
                    rec.get('api_vision_analysis'),    # Save GPT-4 analysis
                    json.dumps(rec.get('analysis_keywords') or []),
                    json.dumps(rec.get('metadata_json') or {})
                ))
                conn.commit()
                conn.close()
                return rec_id
        except Exception as e:
            logger.error(f"Error storing server screenshot: {e}")
            return None

    def get_server_screenshot(self, rec_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM server_screenshots WHERE id = ?', (rec_id,))
                row = cursor.fetchone()
                if not row:
                    conn.close()
                    return None
                columns = [d[0] for d in cursor.description]
                rec = dict(zip(columns, row))
                try:
                    if rec.get('analysis_keywords'):
                        rec['analysis_keywords'] = json.loads(rec['analysis_keywords'])
                    if rec.get('metadata_json'):
                        rec['metadata_json'] = json.loads(rec['metadata_json'])
                except Exception:
                    pass
                conn.close()
                return rec
        except Exception as e:
            logger.error(f"Error getting server screenshot {rec_id}: {e}")
            return None

    def delete_server_screenshot(self, rec_id: str) -> bool:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('DELETE FROM server_screenshots WHERE id = ?', (rec_id,))
                conn.commit()
                success = cursor.rowcount > 0
                conn.close()
                return success
        except Exception as e:
            logger.error(f"Error deleting server screenshot {rec_id}: {e}")
            return False

    def search_server_screenshots(self,
                                  camera_id: Optional[str] = None,
                                  date_start: Optional[str] = None,
                                  date_end: Optional[str] = None,
                                  keyword: Optional[str] = None,
                                  limit: int = 50,
                                  offset: int = 0) -> List[Dict[str, Any]]:
        try:
            with self.lock:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                query = 'SELECT * FROM server_screenshots WHERE 1=1'
                params: List[Any] = []
                if camera_id:
                    query += ' AND camera_id = ?'
                    params.append(camera_id)
                if date_start:
                    query += ' AND created_at >= ?'
                    params.append(date_start)
                if date_end:
                    query += ' AND created_at <= ?'
                    params.append(date_end)
                if keyword:
                    # Match in scene_description or analysis_keywords JSON text
                    query += ' AND (scene_description LIKE ? OR analysis_keywords LIKE ? OR metadata_json LIKE ?)'
                    like = f"%{keyword}%"
                    params.extend([like, like, like])
                query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
                params.extend([int(limit), int(offset)])
                cursor.execute(query, params)
                rows = cursor.fetchall()
                columns = [d[0] for d in cursor.description]
                recs: List[Dict[str, Any]] = []
                for row in rows:
                    r = dict(zip(columns, row))
                    try:
                        if r.get('analysis_keywords'):
                            r['analysis_keywords'] = json.loads(r['analysis_keywords'])
                        if r.get('metadata_json'):
                            r['metadata_json'] = json.loads(r['metadata_json'])
                    except Exception:
                        pass
                    recs.append(r)
                conn.close()
                return recs
        except Exception as e:
            logger.error(f"Error searching server screenshots: {e}")
            return []