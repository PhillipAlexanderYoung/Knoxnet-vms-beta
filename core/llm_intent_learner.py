"""
LLM Intent Learner - Pattern Recognition System

Learns from successful tool executions to help offline/small models
understand user intent without requiring large context windows.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class IntentPattern:
    """Learned intent pattern from successful execution"""
    id: str
    user_query: str  # Original user query
    normalized_query: str  # Normalized version for matching
    tool: str  # Tool that was executed
    parameters: Dict[str, Any]  # Parameters used
    success_count: int  # How many times this pattern succeeded
    last_used: str  # ISO timestamp
    created_at: str  # ISO timestamp
    feedback_score: float  # 0.0-1.0 (user feedback or success rate)
    keywords: List[str]  # Extracted keywords for faster matching


class IntentLearner:
    """Learn and recall successful tool execution patterns"""
    
    def __init__(self, storage_path: str = "data/llm_intent_patterns.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns: Dict[str, IntentPattern] = {}
        self._load_patterns()
    
    def _load_patterns(self):
        """Load learned patterns from disk"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.patterns = {
                        k: IntentPattern(**v) for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self.patterns)} intent patterns")
            except Exception as e:
                logger.error(f"Failed to load intent patterns: {e}")
                self.patterns = {}
        else:
            logger.info("No existing intent patterns, starting fresh")
    
    def _save_patterns(self):
        """Save patterns to disk"""
        try:
            data = {k: asdict(v) for k, v in self.patterns.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.patterns)} intent patterns")
        except Exception as e:
            logger.error(f"Failed to save intent patterns: {e}")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for matching"""
        # Convert to lowercase
        normalized = query.lower().strip()
        
        # Remove punctuation
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove extra whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'can', 'could', 'should', 'may', 'might', 'must',
            'me', 'my', 'you', 'your', 'for', 'to', 'at', 'in', 'on',
            'and', 'or', 'but', 'if', 'then', 'please'
        }
        
        words = self._normalize_query(query).split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def learn_from_execution(
        self,
        user_query: str,
        tool: str,
        parameters: Dict[str, Any],
        success: bool = True,
        feedback_score: float = 1.0
    ):
        """
        Learn from a successful tool execution
        
        Args:
            user_query: Original user query
            tool: Tool that was executed
            parameters: Parameters used
            success: Whether execution was successful
            feedback_score: User feedback (0.0-1.0)
        """
        if not success or not user_query or not tool:
            return
        
        normalized = self._normalize_query(user_query)
        keywords = self._extract_keywords(user_query)
        
        # Create pattern ID
        pattern_id = f"{tool}_{normalized[:50].replace(' ', '_')}"
        
        # Check if pattern exists
        if pattern_id in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_id]
            pattern.success_count += 1
            pattern.last_used = datetime.now().isoformat()
            pattern.feedback_score = (pattern.feedback_score + feedback_score) / 2
            logger.info(f"Updated pattern: {pattern_id} (count: {pattern.success_count})")
        else:
            # Create new pattern
            pattern = IntentPattern(
                id=pattern_id,
                user_query=user_query,
                normalized_query=normalized,
                tool=tool,
                parameters=parameters,
                success_count=1,
                last_used=datetime.now().isoformat(),
                created_at=datetime.now().isoformat(),
                feedback_score=feedback_score,
                keywords=keywords
            )
            self.patterns[pattern_id] = pattern
            logger.info(f"Learned new pattern: {pattern_id}")
        
        self._save_patterns()
    
    def find_matching_pattern(self, user_query: str, threshold: float = 0.7) -> Optional[IntentPattern]:
        """
        Find a matching learned pattern
        
        Args:
            user_query: User's query
            threshold: Similarity threshold (0.0-1.0)
        
        Returns:
            Best matching pattern or None
        """
        if not self.patterns:
            return None
        
        normalized = self._normalize_query(user_query)
        keywords = set(self._extract_keywords(user_query))
        
        best_match = None
        best_score = 0.0
        
        for pattern in self.patterns.values():
            # Calculate similarity score
            score = 0.0
            
            # Exact match (highest score)
            if pattern.normalized_query == normalized:
                score = 1.0
            else:
                # Keyword matching
                pattern_keywords = set(pattern.keywords)
                if keywords and pattern_keywords:
                    intersection = keywords & pattern_keywords
                    union = keywords | pattern_keywords
                    jaccard = len(intersection) / len(union) if union else 0.0
                    score = jaccard
                
                # Substring matching (bonus)
                if normalized in pattern.normalized_query or pattern.normalized_query in normalized:
                    score += 0.2
            
            # Weight by success count and feedback
            weighted_score = score * (1 + min(pattern.success_count / 10, 0.5)) * pattern.feedback_score
            
            if weighted_score > best_score and weighted_score >= threshold:
                best_score = weighted_score
                best_match = pattern
        
        if best_match:
            logger.info(f"Found matching pattern: {best_match.id} (score: {best_score:.2f})")
        
        return best_match
    
    def get_compact_tool_guide(self, max_examples: int = 5) -> str:
        """
        Get a compact guide of common patterns for LLM context
        
        Returns compact string with top patterns for small model context
        """
        if not self.patterns:
            return ""
        
        # Sort by success count and feedback score
        sorted_patterns = sorted(
            self.patterns.values(),
            key=lambda p: p.success_count * p.feedback_score,
            reverse=True
        )
        
        examples = []
        for pattern in sorted_patterns[:max_examples]:
            examples.append(
                f'"{pattern.user_query}" → {pattern.tool}({", ".join(f"{k}={v}" for k, v in pattern.parameters.items())})'
            )
        
        if examples:
            return "Learned patterns:\n" + "\n".join(examples)
        return ""
    
    def get_stats(self) -> Dict[str, Any]:
        """Get learner statistics"""
        if not self.patterns:
            return {"total_patterns": 0}
        
        total_executions = sum(p.success_count for p in self.patterns.values())
        avg_feedback = sum(p.feedback_score for p in self.patterns.values()) / len(self.patterns)
        
        return {
            "total_patterns": len(self.patterns),
            "total_executions": total_executions,
            "avg_feedback_score": round(avg_feedback, 2),
            "top_patterns": [
                {
                    "query": p.user_query,
                    "tool": p.tool,
                    "count": p.success_count
                }
                for p in sorted(
                    self.patterns.values(),
                    key=lambda x: x.success_count,
                    reverse=True
                )[:5]
            ]
        }
    
    def clear_low_performing_patterns(self, min_score: float = 0.3):
        """Remove patterns with low feedback scores"""
        to_remove = [
            pid for pid, pattern in self.patterns.items()
            if pattern.feedback_score < min_score and pattern.success_count < 3
        ]
        
        for pid in to_remove:
            del self.patterns[pid]
        
        if to_remove:
            logger.info(f"Removed {len(to_remove)} low-performing patterns")
            self._save_patterns()
    
    def get_all_patterns(self) -> List[Dict[str, Any]]:
        """Get all patterns as list of dicts"""
        return [asdict(p) for p in self.patterns.values()]
    
    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete a specific pattern"""
        if pattern_id in self.patterns:
            del self.patterns[pattern_id]
            self._save_patterns()
            logger.info(f"Deleted pattern: {pattern_id}")
            return True
        return False
    
    def update_pattern_score(self, pattern_id: str, new_score: float) -> bool:
        """Update feedback score for a pattern"""
        if pattern_id in self.patterns:
            self.patterns[pattern_id].feedback_score = max(0.0, min(1.0, new_score))
            self._save_patterns()
            logger.info(f"Updated pattern {pattern_id} score to {new_score}")
            return True
        return False
    
    def clear_all_patterns(self):
        """Delete all learned patterns"""
        count = len(self.patterns)
        self.patterns = {}
        self._save_patterns()
        logger.warning(f"Cleared all {count} patterns")
        return count

