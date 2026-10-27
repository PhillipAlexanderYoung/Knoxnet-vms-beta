import logging
import json
import requests
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import time

from core.email_client import EmailClient

logger = logging.getLogger(__name__)


class AlertSystem:
    def __init__(self):
        self.alert_rules = {}
        self.alert_history = []
        self.notification_config = {}
        self.load_configuration()

        # Alert cooldown to prevent spam
        self.alert_cooldowns = {}
        self.default_cooldown = 300  # 5 minutes

    def load_configuration(self):
        """Load alert system configuration"""
        try:
            with open('data/alert_config.json', 'r') as f:
                config = json.load(f)
                self.alert_rules = config.get('rules', {})
                self.notification_config = config.get('notifications', {})
            logger.info("Alert configuration loaded successfully")
        except FileNotFoundError:
            logger.info("No alert configuration found, using defaults")
            self.create_default_config()
        except Exception as e:
            logger.error(f"Error loading alert configuration: {e}")
            self.create_default_config()

    def create_default_config(self):
        """Create default alert configuration"""
        self.alert_rules = {
            "person_detection": {
                "enabled": True,
                "priority": "medium",
                "cooldown": 60,
                "conditions": {
                    "confidence_threshold": 0.7,
                    "min_detection_size": 50
                }
            },
            "intrusion": {
                "enabled": True,
                "priority": "high",
                "cooldown": 30,
                "conditions": {
                    "confidence_threshold": 0.8
                }
            },
            "vehicle_detection": {
                "enabled": True,
                "priority": "low",
                "cooldown": 120,
                "conditions": {
                    "confidence_threshold": 0.6
                }
            },
            "crowd_detection": {
                "enabled": True,
                "priority": "medium",
                "cooldown": 180,
                "conditions": {
                    "person_threshold": 5
                }
            }
        }

        self.notification_config = {
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "username": "",
                "password": "",
                "recipients": []
            },
            "webhook": {
                "enabled": False,
                "url": "",
                "headers": {}
            },
            "sms": {
                "enabled": False,
                "api_key": "",
                "service": "twilio",
                "recipients": []
            }
        }

        self.save_configuration()

    def save_configuration(self):
        """Save alert configuration to file"""
        try:
            config = {
                "rules": self.alert_rules,
                "notifications": self.notification_config
            }
            with open('data/alert_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            logger.info("Alert configuration saved")
        except Exception as e:
            logger.error(f"Error saving alert configuration: {e}")

    def check_alerts(self, camera_id: str, analysis_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check analysis results against alert rules"""
        alerts = []

        try:
            detections = analysis_result.get('detections', [])
            timestamp = datetime.now()

            # Check person detection alerts
            person_detections = [d for d in detections if d['class'] == 'person']
            if person_detections and self.should_trigger_alert('person_detection', camera_id):
                for detection in person_detections:
                    if self.meets_detection_criteria('person_detection', detection):
                        alert = self.create_alert(
                            'person_detection',
                            camera_id,
                            f"Person detected with {detection['confidence']:.1%} confidence",
                            detection,
                            timestamp
                        )
                        alerts.append(alert)

            # Check vehicle detection alerts
            vehicle_classes = ['car', 'truck', 'motorcycle', 'bus']
            vehicle_detections = [d for d in detections if d['class'] in vehicle_classes]
            if vehicle_detections and self.should_trigger_alert('vehicle_detection', camera_id):
                for detection in vehicle_detections:
                    if self.meets_detection_criteria('vehicle_detection', detection):
                        alert = self.create_alert(
                            'vehicle_detection',
                            camera_id,
                            f"{detection['class'].title()} detected with {detection['confidence']:.1%} confidence",
                            detection,
                            timestamp
                        )
                        alerts.append(alert)

            # Check crowd detection
            if len(person_detections) > 0 and self.should_trigger_alert('crowd_detection', camera_id):
                person_count = len(person_detections)
                threshold = self.alert_rules['crowd_detection']['conditions']['person_threshold']

                if person_count >= threshold:
                    alert = self.create_alert(
                        'crowd_detection',
                        camera_id,
                        f"Crowd detected: {person_count} persons",
                        {'person_count': person_count, 'detections': person_detections},
                        timestamp
                    )
                    alerts.append(alert)

            # Check custom alerts from analysis
            for custom_alert in analysis_result.get('alerts', []):
                alert_type = custom_alert.get('type', 'custom')
                if self.should_trigger_alert(alert_type, camera_id):
                    alert = self.create_alert(
                        alert_type,
                        camera_id,
                        custom_alert.get('description', 'Custom alert triggered'),
                        custom_alert,
                        timestamp
                    )
                    alerts.append(alert)

            # Update cooldowns for triggered alerts
            for alert in alerts:
                self.update_cooldown(alert['type'], camera_id)

        except Exception as e:
            logger.error(f"Error checking alerts for camera {camera_id}: {e}")

        return alerts

    def should_trigger_alert(self, alert_type: str, camera_id: str) -> bool:
        """Check if alert should be triggered based on rules and cooldowns"""
        try:
            # Check if alert type is enabled
            if alert_type not in self.alert_rules:
                return False

            if not self.alert_rules[alert_type].get('enabled', True):
                return False

            # Check cooldown
            cooldown_key = f"{alert_type}:{camera_id}"
            if cooldown_key in self.alert_cooldowns:
                last_triggered = self.alert_cooldowns[cooldown_key]
                cooldown_period = self.alert_rules[alert_type].get('cooldown', self.default_cooldown)

                if (datetime.now() - last_triggered).total_seconds() < cooldown_period:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking alert trigger for {alert_type}: {e}")
            return False

    def meets_detection_criteria(self, alert_type: str, detection: Dict[str, Any]) -> bool:
        """Check if detection meets alert criteria"""
        try:
            if alert_type not in self.alert_rules:
                return False

            conditions = self.alert_rules[alert_type].get('conditions', {})

            # Check confidence threshold
            confidence_threshold = conditions.get('confidence_threshold', 0.5)
            if detection['confidence'] < confidence_threshold:
                return False

            # Check minimum detection size
            min_size = conditions.get('min_detection_size', 0)
            if min_size > 0:
                bbox = detection['bbox']
                detection_area = bbox['width'] * bbox['height']
                if detection_area < min_size:
                    return False

            return True

        except Exception as e:
            logger.error(f"Error checking detection criteria: {e}")
            return False

    def create_alert(self, alert_type: str, camera_id: str, description: str,
                     data: Dict[str, Any], timestamp: datetime) -> Dict[str, Any]:
        """Create alert object"""
        alert_id = f"{alert_type}_{camera_id}_{int(timestamp.timestamp())}"
        priority = self.alert_rules.get(alert_type, {}).get('priority', 'medium')

        alert = {
            'id': alert_id,
            'type': alert_type,
            'camera_id': camera_id,
            'timestamp': timestamp.isoformat(),
            'priority': priority,
            'description': description,
            'data': data,
            'acknowledged': False,
            'resolved': False
        }

        # Add to history
        self.alert_history.append(alert)

        # Keep only last 1000 alerts
        if len(self.alert_history) > 1000:
            self.alert_history = self.alert_history[-1000:]

        logger.info(f"Alert created: {alert_id} - {description}")

        return alert

    def _emit_cloud_event(self, alert: Dict[str, Any]) -> None:
        """Cloud event upload is intentionally disabled in the public beta."""
        _ = alert
        return

    def update_cooldown(self, alert_type: str, camera_id: str):
        """Update cooldown timestamp for alert type and camera"""
        cooldown_key = f"{alert_type}:{camera_id}"
        self.alert_cooldowns[cooldown_key] = datetime.now()

    def send_notifications(self, alert: Dict[str, Any]):
        """Send notifications for alert"""
        try:
            # Send email notification
            if self.notification_config.get('email', {}).get('enabled', False):
                threading.Thread(
                    target=self.send_email_notification,
                    args=(alert,),
                    daemon=True
                ).start()

            # Send webhook notification
            if self.notification_config.get('webhook', {}).get('enabled', False):
                threading.Thread(
                    target=self.send_webhook_notification,
                    args=(alert,),
                    daemon=True
                ).start()

            # Send SMS notification
            if self.notification_config.get('sms', {}).get('enabled', False):
                threading.Thread(
                    target=self.send_sms_notification,
                    args=(alert,),
                    daemon=True
                ).start()

        except Exception as e:
            logger.error(f"Error sending notifications for alert {alert['id']}: {e}")

    def send_email_notification(self, alert: Dict[str, Any]):
        """Send email notification"""
        try:
            email_config = self.notification_config['email']
            client = EmailClient.from_dict(email_config)
            if not client:
                logger.warning("Email notification skipped: email client not configured.")
                return

            subject = f"Security Alert: {alert['type'].replace('_', ' ').title()}"
            body = (
                "Security Alert Notification\n\n"
                f"Alert ID: {alert['id']}\n"
                f"Type: {alert['type'].replace('_', ' ').title()}\n"
                f"Camera: {alert['camera_id']}\n"
                f"Priority: {alert['priority'].upper()}\n"
                f"Time: {alert['timestamp']}\n"
                f"Description: {alert['description']}\n\n"
                "Please review the security dashboard for more details."
            )

            recipients = email_config.get('recipients')
            client.send(subject=subject, body=body, to=recipients)
            logger.info(f"Email notification sent for alert {alert['id']}")
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")

    def send_webhook_notification(self, alert: Dict[str, Any]):
        """Send webhook notification"""
        try:
            webhook_config = self.notification_config['webhook']
            url = webhook_config.get('url')

            if not url:
                return

            headers = webhook_config.get('headers', {})
            headers['Content-Type'] = 'application/json'

            payload = {
                'alert': alert,
                'timestamp': datetime.now().isoformat(),
                'source': 'knoxnet_dashboard'
            }

            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()

            logger.info(f"Webhook notification sent for alert {alert['id']}")

        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")

    def send_sms_notification(self, alert: Dict[str, Any]):
        """Send SMS notification (placeholder - would integrate with SMS service)"""
        try:
            sms_config = self.notification_config['sms']

            if not sms_config.get('recipients'):
                return

            message = f"Security Alert: {alert['description']} at {alert['camera_id']}"

            # This would integrate with actual SMS service (Twilio, etc.)
            logger.info(f"SMS notification would be sent for alert {alert['id']}: {message}")

        except Exception as e:
            logger.error(f"Error sending SMS notification: {e}")

    def get_alerts(self, limit: int = 100, priority: str = None,
                   camera_id: str = None) -> List[Dict[str, Any]]:
        """Get alerts with optional filtering"""
        try:
            filtered_alerts = self.alert_history.copy()

            # Filter by priority
            if priority:
                filtered_alerts = [a for a in filtered_alerts if a['priority'] == priority]

            # Filter by camera
            if camera_id:
                filtered_alerts = [a for a in filtered_alerts if a['camera_id'] == camera_id]

            # Sort by timestamp (newest first)
            filtered_alerts.sort(key=lambda x: x['timestamp'], reverse=True)

            return filtered_alerts[:limit]

        except Exception as e:
            logger.error(f"Error getting alerts: {e}")
            return []

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.alert_history:
                if alert['id'] == alert_id:
                    alert['acknowledged'] = True
                    alert['acknowledged_at'] = datetime.now().isoformat()
                    logger.info(f"Alert acknowledged: {alert_id}")
                    return True
            return False

        except Exception as e:
            logger.error(f"Error acknowledging alert {alert_id}: {e}")
            return False

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            for alert in self.alert_history:
                if alert['id'] == alert_id:
                    alert['resolved'] = True
                    alert['resolved_at'] = datetime.now().isoformat()
                    logger.info(f"Alert resolved: {alert_id}")
                    return True
            return False

        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
            return False