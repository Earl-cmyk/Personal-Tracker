"""
Personal Metrics Dashboard - Local-First Application
Architecture: Modular adapters with clean separation of concerns
"""

import json
import sqlite3
import threading
import logging
import os
import asyncio
import random
import re
import secrets
import hashlib
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from urllib.parse import urlencode
import bcrypt
from flask import request
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import ollama
import requests
from bs4 import BeautifulSoup
from cryptography.fernet import Fernet
from apscheduler.schedulers.background import BackgroundScheduler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the base directory
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = hashlib.sha256(b'personal_dashboard_local_key').hexdigest()
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATA_DIR / "dashboard.db"}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# ============================================================================
# DATA MODELS
# ============================================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    name = db.Column(db.String(100), nullable=False, default='User')
    password_hash = db.Column(db.String(200))  # Hashed password
    google_id = db.Column(db.String(200), unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    settings = db.Column(db.Text, default='{"theme": "night"}')
    
    # Relationships
    app_connections = db.relationship('AppConnection', backref='user', lazy=True, cascade='all, delete-orphan')
    notifications = db.relationship('Notification', backref='user', lazy=True)
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def check_password(self, password):
        """Check password against hash"""
        if not self.password_hash:
            return False
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

class AppConnection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    app_name = db.Column(db.String(100), nullable=False)
    platform = db.Column(db.String(100), nullable=False)
    profile_url = db.Column(db.Text)  # Store profile URL
    credentials = db.Column(db.Text, default='{}')
    last_sync = db.Column(db.DateTime)
    sync_status = db.Column(db.String(50), default='pending')
    config = db.Column(db.Text, default='{}')
    
    __table_args__ = (db.UniqueConstraint('user_id', 'app_name', name='_user_app_uc'),)


class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    source = db.Column(db.String(100))
    priority = db.Column(db.String(20), default='medium')
    read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Metric(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    connection_id = db.Column(db.Integer, db.ForeignKey('app_connection.id'), nullable=False)
    metric_type = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    label = db.Column(db.String(200))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    
    connection = db.relationship('AppConnection', backref='metrics')


class Activity(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    action = db.Column(db.String(100), nullable=False)
    details = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='activities')

class BrainDump(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    content = db.Column(db.Text)
    analysis = db.Column(db.Text)  # JSON analysis from AI
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = db.relationship('User', backref='brain_dumps')


class CalendarEvent(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    start_time = db.Column(db.DateTime, nullable=False)
    end_time = db.Column(db.DateTime)
    all_day = db.Column(db.Boolean, default=False)
    event_type = db.Column(db.String(50), default='task')  # task, notification, appointment
    priority = db.Column(db.String(20), default='medium')
    source_id = db.Column(db.Integer)  # ID of source (notification_id, etc.)
    source_type = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='calendar_events')


class KnowledgeItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(100), nullable=False)  # health, law, philosophy, science
    title = db.Column(db.String(200), nullable=False)
    content = db.Column(db.Text, nullable=False)
    tags = db.Column(db.Text)  # JSON array of tags
    source = db.Column(db.String(200))
    is_fallback = db.Column(db.Boolean, default=True)  # True if from fallback library
    created_at = db.Column(db.DateTime, default=datetime.utcnow)# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class Platform(Enum):
    """Supported platforms enumeration"""
    GITHUB = "github"
    VERCEL = "vercel"
    YOUTUBE = "youtube"
    TIKTOK = "tiktok"
    SPOTIFY = "spotify"
    PINTEREST = "pinterest"
    LINKEDIN = "linkedin"
    CASHEW = "cashew"
    GOOGLE_CLASSROOM = "google_classroom"
    MYFITNESSPAL = "myfitnesspal"
    CHESS = "chess"
    STUDYPOOL = "studypool"
    TALKPAL = "talkpal"
    CSTIMER = "cstimer"
    LICHESS = "lichess"
    WATTPAD = "wattpad"
    PIXIV = "pixiv"
    BANDLAB = "bandlab"
    GCASH = "gcash"
    WESTERN_UNION = "western_union"
    MARIBANK = "maribank"
    SHOPEE = "shopee"
    LAZADA = "lazada"


@dataclass
class MetricData:
    """Standardized metric data structure"""
    type: str
    value: float
    label: str
    change: Optional[float] = None
    timestamp: datetime = None
    
    def to_dict(self):
        return asdict(self)


# ============================================================================
# CREDENTIAL MANAGER
# ============================================================================

class CredentialManager:
    """Manage encryption/decryption of credentials"""
    
    def __init__(self):
        # Generate or load encryption key
        key_path = DATA_DIR / 'encryption.key'
        if key_path.exists():
            with open(key_path, 'rb') as f:
                self.key = f.read()
        else:
            self.key = Fernet.generate_key()
            with open(key_path, 'wb') as f:
                f.write(self.key)
        
        self.cipher = Fernet(self.key)
    
    def encrypt(self, data: Dict) -> str:
        """Encrypt credentials dictionary"""
        json_str = json.dumps(data)
        encrypted = self.cipher.encrypt(json_str.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt(self, encrypted_str: str) -> Dict:
        """Decrypt credentials"""
        try:
            encrypted = base64.b64decode(encrypted_str.encode())
            decrypted = self.cipher.decrypt(encrypted)
            return json.loads(decrypted.decode())
        except:
            return {}


# Initialize credential manager
credential_manager = CredentialManager()


# ============================================================================
# PUBLIC CONNECTORS (FOR PUBLIC PROFILES)
# ============================================================================

class PublicBaseConnector:
    """Base class for public profile connectors"""
    
    def __init__(self, connection: AppConnection):
        self.connection = connection
        self.config = json.loads(connection.config or '{}')
        self.profile_url = connection.profile_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    async def fetch_metrics(self) -> List[MetricData]:
        """Fetch metrics from public profile"""
        if not self.profile_url:
            return []
        
        try:
            response = self.session.get(self.profile_url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML and extract metrics
            metrics = self.extract_metrics_from_html(response.text)
            
            # Add fallback metrics if extraction fails
            if not metrics:
                metrics = self._get_sample_metrics()
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to fetch {self.connection.platform} data: {e}")
            return self._get_sample_metrics()
    
    def extract_metrics_from_html(self, html: str) -> List[MetricData]:
        """Extract metrics from HTML - to be overridden by subclasses"""
        return []
    
    def _get_sample_metrics(self):
        """Fallback sample metrics"""
        return [
            MetricData(type="views", value=0, label="Views"),
            MetricData(type="followers", value=0, label="Followers")
        ]
    
    async def fetch_notifications(self) -> List[Dict]:
        """Public platforms don't have notifications"""
        return []
    
    async def test_connection(self) -> bool:
        """Test if profile URL is accessible"""
        if not self.profile_url:
            return False
        
        try:
            response = self.session.head(self.profile_url, timeout=5)
            return response.status_code == 200
        except:
            return False


class YouTubePublicConnector(PublicBaseConnector):
    """Scrape public YouTube channel metrics"""
    
    async def fetch_metrics(self) -> List[MetricData]:
        """Fetch YouTube public metrics"""
        if not self.profile_url:
            return []
        
        metrics = []
        
        try:
            # Method 1: Try to get via YouTube Data API (public, no auth needed for basic info)
            channel_id = self._extract_channel_id(self.profile_url)
            
            if channel_id:
                # Try to get via oEmbed first (simpler)
                oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/channel/{channel_id}&format=json"
                oembed_response = self.session.get(oembed_url)
                
                if oembed_response.status_code == 200:
                    oembed_data = oembed_response.json()
                    # Extract from title if possible
                    title = oembed_data.get('title', '')
                    if 'subscribers' in title.lower():
                        sub_match = re.search(r'(\d+(?:\.\d+)?[KM]?)\s+subscribers', title)
                        if sub_match:
                            subscribers = self._parse_count(sub_match.group(1))
                            metrics.append(MetricData(
                                type="subscribers",
                                value=subscribers,
                                label="Subscribers"
                            ))
            
            # Method 2: Scrape from HTML
            response = self.session.get(self.profile_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Look for subscriber count
            sub_elements = soup.find_all(string=re.compile(r'subscribers', re.I))
            for element in sub_elements:
                text = str(element)
                match = re.search(r'(\d+(?:\.\d+)?[KM]?)\s+subscribers', text, re.I)
                if match:
                    subscribers = self._parse_count(match.group(1))
                    metrics.append(MetricData(
                        type="subscribers",
                        value=subscribers,
                        label="Subscribers"
                    ))
                    break
            
            # Look for video count
            video_elements = soup.find_all(string=re.compile(r'videos', re.I))
            for element in video_elements:
                text = str(element)
                match = re.search(r'(\d+(?:\.\d+)?[KM]?)\s+videos', text, re.I)
                if match:
                    videos = self._parse_count(match.group(1))
                    metrics.append(MetricData(
                        type="videos",
                        value=videos,
                        label="Videos"
                    ))
                    break
            
            # Add sample view count
            if metrics:
                subscribers_value = next((m.value for m in metrics if m.type == "subscribers"), 0)
                metrics.append(MetricData(
                    type="estimated_views",
                    value=subscribers_value * 100,
                    label="Estimated Views",
                    change=2.5
                ))
            
        except Exception as e:
            logger.error(f"YouTube scraping error: {e}")
        
        # Fallback to sample metrics
        if not metrics:
            metrics = [
                MetricData(type="subscribers", value=1500, label="Subscribers", change=3.2),
                MetricData(type="videos", value=42, label="Videos", change=5.0),
                MetricData(type="views", value=125000, label="Total Views", change=4.1)
            ]
        
        return metrics
    
    def _extract_channel_id(self, url: str) -> str:
        """Extract channel ID from YouTube URL"""
        patterns = [
            r'youtube\.com/channel/([^/?&]+)',
            r'youtube\.com/c/([^/?&]+)',
            r'youtube\.com/user/([^/?&]+)',
            r'youtube\.com/@([^/?&]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return ""
    
    def _parse_count(self, count_str: str) -> float:
        """Parse counts like 1.2K, 5M into numbers"""
        count_str = count_str.upper().replace(',', '')
        
        if 'K' in count_str:
            return float(count_str.replace('K', '')) * 1000
        elif 'M' in count_str:
            return float(count_str.replace('M', '')) * 1000000
        elif 'B' in count_str:
            return float(count_str.replace('B', '')) * 1000000000
        else:
            try:
                return float(count_str)
            except:
                return 0


class GitHubPublicConnector(PublicBaseConnector):
    """Scrape public GitHub profile metrics"""
    
    async def fetch_metrics(self) -> List[MetricData]:
        """Fetch GitHub public metrics"""
        if not self.profile_url:
            return []
        
        metrics = []
        
        try:
            username = self._extract_username(self.profile_url)
            
            if username:
                # Get user info from GitHub API (public endpoint)
                api_url = f"https://api.github.com/users/{username}"
                response = self.session.get(api_url)
                
                if response.status_code == 200:
                    user_data = response.json()
                    
                    metrics.append(MetricData(
                        type="followers",
                        value=user_data.get('followers', 0),
                        label="Followers",
                        change=1.2
                    ))
                    
                    metrics.append(MetricData(
                        type="following",
                        value=user_data.get('following', 0),
                        label="Following"
                    ))
                    
                    metrics.append(MetricData(
                        type="public_repos",
                        value=user_data.get('public_repos', 0),
                        label="Public Repositories",
                        change=0.5
                    ))
                    
                    # Get contribution data (approximate)
                    events_url = f"https://api.github.com/users/{username}/events/public"
                    events_response = self.session.get(events_url)
                    if events_response.status_code == 200:
                        events = events_response.json()
                        recent_events = [e for e in events if e.get('type') in ['PushEvent', 'CreateEvent']]
                        metrics.append(MetricData(
                            type="recent_activity",
                            value=len(recent_events),
                            label="Recent Activities"
                        ))
        
        except Exception as e:
            logger.error(f"GitHub API error: {e}")
        
        # Fallback
        if not metrics:
            metrics = [
                MetricData(type="followers", value=85, label="Followers", change=1.2),
                MetricData(type="repos", value=12, label="Repositories", change=0.5),
                MetricData(type="stars", value=42, label="Stars Received", change=2.1)
            ]
        
        return metrics
    
    def _extract_username(self, url: str) -> str:
        """Extract GitHub username from URL"""
        patterns = [
            r'github\.com/([^/?&]+)',
            r'github\.com/([^/?&]+)/?$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1).split('/')[0]
        return ""


class TikTokPublicConnector(PublicBaseConnector):
    """Scrape public TikTok metrics"""
    
    async def fetch_metrics(self) -> List[MetricData]:
        """TikTok is tricky to scrape, use fallback or API"""
        return [
            MetricData(type="followers", value=1000, label="Followers", change=5.2),
            MetricData(type="likes", value=5000, label="Total Likes", change=3.1),
            MetricData(type="videos", value=25, label="Videos", change=2.0)
        ]


class ChessPublicConnector(PublicBaseConnector):
    """Scrape Chess.com public stats"""
    
    async def fetch_metrics(self) -> List[MetricData]:
        if not self.profile_url:
            return []
        
        metrics = []
        
        try:
            # Extract username
            username = self.profile_url.split('/')[-1]
            api_url = f"https://api.chess.com/pub/player/{username}/stats"
            response = self.session.get(api_url)
            
            if response.status_code == 200:
                stats = response.json()
                
                # Chess ratings
                for game_type in ['chess_blitz', 'chess_rapid', 'chess_bullet']:
                    if game_type in stats:
                        rating = stats[game_type].get('last', {}).get('rating', 0)
                        metrics.append(MetricData(
                            type=f"{game_type}_rating",
                            value=rating,
                            label=f"{game_type.replace('chess_', '').title()} Rating"
                        ))
                
        except Exception as e:
            logger.error(f"Chess.com error: {e}")
        
        # Fallback
        if not metrics:
            metrics = [
                MetricData(type="blitz_rating", value=1500, label="Blitz Rating", change=12),
                MetricData(type="rapid_rating", value=1600, label="Rapid Rating", change=8),
                MetricData(type="games_played", value=250, label="Games Played", change=5)
            ]
        
        return metrics


class SimpleConnector(PublicBaseConnector):
    """Simple connector for platforms without specific implementation"""
    
    async def fetch_metrics(self) -> List[MetricData]:
        """Return sample metrics for unsupported platforms"""
        return [
            MetricData(type="activity", value=random.randint(10, 100), label="Activity Score", change=random.uniform(-5, 10)),
            MetricData(type="engagement", value=random.randint(100, 1000), label="Engagement", change=random.uniform(0, 15)),
            MetricData(type="updates", value=random.randint(1, 20), label="Recent Updates", change=random.uniform(-2, 8))
        ]


# ============================================================================
# CONNECTOR REGISTRY
# ============================================================================

CONNECTOR_REGISTRY = {
    # Public connectors
    'youtube': YouTubePublicConnector,
    'github': GitHubPublicConnector,
    'tiktok': TikTokPublicConnector,
    'chess': ChessPublicConnector,
    'chess.com': ChessPublicConnector,
    'lichess': ChessPublicConnector,
    
    # Other platforms use simple connector
    'wattpad': SimpleConnector,
    'pinterest': SimpleConnector,
    'pixiv': SimpleConnector,
    'spotify': SimpleConnector,
    'linkedin': SimpleConnector,
    'cashew': SimpleConnector,
    'google_classroom': SimpleConnector,
    'myfitnesspal': SimpleConnector,
    'studypool': SimpleConnector,
    'talkpal': SimpleConnector,
    'cstimer': SimpleConnector,
    'bandlab': SimpleConnector,
    'gcash': SimpleConnector,
    'western_union': SimpleConnector,
    'maribank': SimpleConnector,
    'shopee': SimpleConnector,
    'lazada': SimpleConnector,
    'vercel': SimpleConnector,
}


# ============================================================================
# AI SUMMARIZER (WITH FALLBACKS)
# ============================================================================

class NotificationSummarizer:
    """Local LLM-based notification summarizer with fallbacks"""
    
    def __init__(self):
        self.ollama_available = False
        self.ollama_client = None
        self.model = "llama2"
        
        # Try to initialize Ollama
        try:
            self.ollama_client = ollama.Client(host='http://localhost:11434')
            # Test connection
            self.ollama_client.list()
            self.ollama_available = True
            logger.info("Ollama connected successfully")
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. Using fallback insights.")
            self.ollama_available = False
    
    def summarize_notifications(self, notifications: List[Dict]) -> str:
        """Summarize multiple notifications into a concise update"""
        
        if not notifications:
            return "No new notifications"
        
        if self.ollama_available:
            # Try Ollama summarization
            notifications_text = "\n".join(
                [f"- {n.get('title', '')}: {n.get('content', '')}" 
                 for n in notifications[:5]]
            )
            
            prompt = f"""Summarize these notifications into a concise daily update (max 3 sentences):
            
            {notifications_text}
            
            Summary:"""
            
            try:
                response = self.ollama_client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={'temperature': 0.3, 'max_tokens': 150}
                )
                return response['response'].strip()
            except Exception as e:
                logger.error(f"Ollama summarization failed: {e}")
                # Fall through to basic summary
        
        # Basic fallback summary
        app_names = list(set([n.get('source', 'Unknown') for n in notifications]))
        if len(app_names) > 3:
            app_text = f"{len(app_names)} different apps"
        else:
            app_text = ", ".join(app_names)
        
        summaries = [
            f"You have {len(notifications)} updates from {app_text}.",
            f"New activity detected across {len(app_names)} platforms.",
            f"Summary of recent updates from your connected apps.",
            f"Here's what's happening with your connected services."
        ]
        
        return random.choice(summaries)
    
    def generate_daily_insight(self, category: str, topic: str = "") -> str:
        """Generate daily insights for knowledge feeds with fallbacks"""
        
        if self.ollama_available:
            prompts = {
                "health": "Provide one brief, evidence-based health tip from medical professionals today:",
                "law": "Provide one important update or fact about Philippine law today:",
                "philosophy": "Share one meaningful philosophy quote with brief context today:",
                "bible": "Share one Bible verse with brief practical application today:"
            }
            
            if category not in prompts:
                return f"No insight available for {category}"
            
            prompt = prompts[category]
            
            try:
                response = self.ollama_client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={'temperature': 0.7, 'max_tokens': 100}
                )
                return response['response'].strip()
            except Exception as e:
                logger.error(f"Ollama insight generation failed: {e}")
                # Fall through to static insights
        
        # Static fallback insights
        fallback_insights = {
            "health": self._get_health_insight(),
            "law": self._get_law_insight(),
            "philosophy": self._get_philosophy_insight(),
            "bible": self._get_bible_insight()
        }
        
        return fallback_insights.get(category, "Daily insight will appear here.")
    
    def _get_health_insight(self):
        """Health tip fallbacks"""
        tips = [
            "Stay hydrated! Aim for 8 glasses of water daily for optimal health.",
            "Get 7-9 hours of sleep each night to support brain function and overall health.",
            "Incorporate 30 minutes of moderate exercise into your daily routine.",
            "Practice mindfulness meditation for 10 minutes daily to reduce stress.",
            "Eat a rainbow of fruits and vegetables for diverse nutrients.",
            "Take regular breaks from screens to protect your eye health.",
            "Maintain good posture to prevent back and neck pain.",
            "Wash your hands regularly to prevent illness transmission."
        ]
        return random.choice(tips)
    
    def _get_law_insight(self):
        """Philippine law fallbacks"""
        laws = [
            "The 1987 Philippine Constitution guarantees freedom of speech under Article III, Section 4.",
            "Republic Act 9262, the Anti-Violence Against Women and Their Children Act, provides protection orders.",
            "The Data Privacy Act of 2012 (RA 10173) protects personal information in processing systems.",
            "Under Philippine labor law, regular employees are entitled to 13th month pay.",
            "The Cybercrime Prevention Act of 2012 (RA 10175) addresses computer-related offenses.",
            "The Philippine Clean Air Act (RA 8749) aims to achieve and maintain clean air.",
            "The Indigenous Peoples Rights Act (RA 8371) recognizes and promotes indigenous peoples' rights.",
            "The Philippine Fisheries Code (RA 8550) provides for the sustainable development of fisheries."
        ]
        return random.choice(laws)
    
    def _get_philosophy_insight(self):
        """Philosophy quote fallbacks"""
        quotes = [
            "\"The unexamined life is not worth living.\" - Socrates",
            "\"I think, therefore I am.\" - René Descartes",
            "\"The only thing I know is that I know nothing.\" - Socrates",
            "\"Happiness is the highest good.\" - Aristotle",
            "\"Man is condemned to be free.\" - Jean-Paul Sartre",
            "\"God is dead.\" - Friedrich Nietzsche",
            "\"The mind is furnished with ideas by experience alone.\" - John Locke",
            "\"To be is to be perceived.\" - George Berkeley"
        ]
        return random.choice(quotes)
    
    def _get_bible_insight(self):
        """Bible verse fallbacks"""
        verses = [
            "\"I can do all things through Christ who strengthens me.\" - Philippians 4:13",
            "\"For God so loved the world that he gave his only Son.\" - John 3:16",
            "\"The Lord is my shepherd; I shall not want.\" - Psalm 23:1",
            "\"Be still, and know that I am God.\" - Psalm 46:10",
            "\"Trust in the Lord with all your heart.\" - Proverbs 3:5",
            "\"Love your neighbor as yourself.\" - Mark 12:31",
            "\"Do not be anxious about anything.\" - Philippians 4:6",
            "\"The fruit of the Spirit is love, joy, peace.\" - Galatians 5:22"
        ]
        return random.choice(verses)


# ============================================================================
# DASHBOARD MANAGER
# ============================================================================

class DashboardManager:
    """Manages dashboard operations and data aggregation"""
    
    def __init__(self, app_context):
        self.app = app_context
        self.summarizer = NotificationSummarizer()
        self.scheduler = None
    
    def start(self):
        """Start the scheduler"""
        if self.scheduler is None:
            self.scheduler = BackgroundScheduler()
            self.setup_scheduler()
            self.scheduler.start()
            logger.info("Dashboard scheduler started")
    
    def setup_scheduler(self):
        """Setup scheduled tasks"""
        # Daily sync at 6 AM
        self.scheduler.add_job(
            func=self.sync_all_connections,
            trigger='cron',
            hour=6,
            minute=0,
            id='daily_sync'
        )
        
        # Hourly notification check
        self.scheduler.add_job(
            func=self.check_notifications,
            trigger='interval',
            hours=1,
            id='hourly_check'
        )
    
    def sync_all_connections(self):
        """Sync all user connections"""
        with self.app.app_context():
            connections = AppConnection.query.filter_by(sync_status='active').all()
            
            for connection in connections:
                asyncio.run(self.sync_connection(connection))
    
    async def sync_connection(self, connection: AppConnection):
        """Sync a single connection"""
        connector_class = CONNECTOR_REGISTRY.get(connection.platform)
        if not connector_class:
            logger.warning(f"No connector found for platform: {connection.platform}")
            return
        
        connector = connector_class(connection)
        
        try:
            # Fetch metrics
            metrics = await connector.fetch_metrics()
            
            # Store metrics
            for metric_data in metrics:
                metric = Metric(
                    connection_id=connection.id,
                    metric_type=metric_data.type,
                    value=metric_data.value,
                    label=metric_data.label,
                    timestamp=metric_data.timestamp or datetime.utcnow()
                )
                db.session.add(metric)
            
            # Fetch and summarize notifications
            raw_notifications = await connector.fetch_notifications()
            if raw_notifications:
                summary = self.summarizer.summarize_notifications(raw_notifications)
                
                # Create notification entry
                notification = Notification(
                    user_id=connection.user_id,
                    title=f"{connection.app_name} Update",
                    content=summary,
                    source=connection.platform,
                    priority="medium"
                )
                db.session.add(notification)
            
            connection.last_sync = datetime.utcnow()
            connection.sync_status = 'synced'
            db.session.commit()
            logger.info(f"Synced {connection.app_name}")
            
        except Exception as e:
            logger.error(f"Sync failed for {connection.app_name}: {e}")
            connection.sync_status = 'error'
            db.session.commit()
    
    async def check_notifications(self):
        """Check and summarize notifications"""
        logger.info("Checking for notifications...")
    
    def stop(self):
        """Stop the scheduler"""
        if self.scheduler:
            self.scheduler.shutdown()
            self.scheduler = None


# ============================================================================
# TEMPLATE FILTERS AND HELPERS
# ============================================================================

@app.template_filter('time_ago')
def time_ago_filter(dt):
    if not dt:
        return "Never"
    
    now = datetime.utcnow()
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt.replace('Z', '+00:00'))
        except:
            return "Unknown"
    
    diff = now - dt
    
    if diff.days > 365:
        return f"{diff.days // 365}y ago"
    elif diff.days > 30:
        return f"{diff.days // 30}mo ago"
    elif diff.days > 0:
        return f"{diff.days}d ago"
    elif diff.seconds > 3600:
        return f"{diff.seconds // 3600}h ago"
    elif diff.seconds > 60:
        return f"{diff.seconds // 60}m ago"
    else:
        return "Just now"


def get_app_icon(platform_name):
    """Get appropriate Font Awesome icon for platform"""
    icon_map = {
        # Writing & Creative
        'wattpad': 'fas fa-book',
        'pinterest': 'fab fa-pinterest',
        'pixiv': 'fas fa-palette',
        'tiktok': 'fab fa-tiktok',
        'youtube': 'fab fa-youtube',
        'bandlab': 'fas fa-music',
        'spotify': 'fab fa-spotify',
        
        # Coding & Tech
        'github': 'fab fa-github',
        'vercel': 'fas fa-cloud',
        
        # Student/Work
        'google_classroom': 'fas fa-graduation-cap',
        'studypool': 'fas fa-book-open',
        'linkedin': 'fab fa-linkedin',
        
        # Finance
        'cashew': 'fas fa-wallet',
        'gcash': 'fas fa-money-bill-wave',
        'western_union': 'fas fa-globe-americas',
        'maribank': 'fas fa-university',
        'shopee': 'fas fa-shopping-bag',
        'lazada': 'fas fa-shopping-cart',
        
        # Productivity
        'myfitnesspal': 'fas fa-dumbbell',
        
        # Skills & Hobbies
        'talkpal': 'fas fa-language',
        'cstimer': 'fas fa-stopwatch',
        'chess': 'fas fa-chess',
        'lichess': 'fas fa-chess-board',
        
        # Default fallback
        'default': 'fas fa-cube'
    }
    
    if not platform_name:
        return icon_map['default']
    
    key = str(platform_name).lower().strip()
    
    # Check exact match
    if key in icon_map:
        return icon_map[key]
    
    # Check partial matches
    for platform_key, icon in icon_map.items():
        if key in platform_key or platform_key in key:
            return icon
    
    return icon_map['default']


@app.context_processor
def utility_processor():
    """Make utility functions available in templates"""
    return {
        'get_app_icon': get_app_icon,
        'time_ago': time_ago_filter,
        'now': datetime.utcnow
    }


# ============================================================================
# ROUTES
# ============================================================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@app.route('/')
@login_required
def dashboard():
    """Main dashboard view - renamed from index"""
    connections = AppConnection.query.filter_by(user_id=current_user.id).all()
    
    # Get recent notifications
    notifications = Notification.query.filter_by(
        user_id=current_user.id
    ).order_by(Notification.created_at.desc()).limit(10).all()
    
    # Get daily insights
    insights = get_daily_insights()
    
    # Get metrics for each connection
    connection_metrics = {}
    total_metrics_count = 0
    
    for connection in connections:
        metrics = Metric.query.filter_by(connection_id=connection.id)\
            .order_by(Metric.timestamp.desc()).limit(5).all()
        
        connection_metrics[connection.id] = [
            {
                'type': m.metric_type,
                'value': m.value,
                'label': m.label,
                'timestamp': m.timestamp
            } for m in metrics
        ]
        
        total_metrics_count += len(metrics)
    
    # Calculate stats
    active_connections = len([c for c in connections if c.sync_status in ['active', 'synced']])
    unread_notifications = len([n for n in notifications if not n.read])
    
    return render_template('dashboard.html', 
                         connections=connections,
                         notifications=notifications,
                         insights=insights,
                         connection_metrics=connection_metrics,
                         total_metrics_count=total_metrics_count,
                         active_connections=active_connections,
                         unread_notifications=unread_notifications,
                         total_connections=len(connections))


@app.route('/add-app', methods=['GET', 'POST'])
@login_required
def add_app():
    """Add new app/connection with profile URL"""
    
    if request.method == 'POST':
        app_name = request.form.get('app_name')
        platform = request.form.get('platform')
        profile_url = request.form.get('profile_url', '').strip()
        
        # Check if app already exists
        existing = AppConnection.query.filter_by(
            user_id=current_user.id,
            app_name=app_name
        ).first()
        
        if existing:
            flash(f'App "{app_name}" already exists!', 'error')
            return redirect(url_for('add_app'))
        
        # Validate URL for platforms that need it
        platforms_needing_url = ['youtube', 'github', 'tiktok', 'chess', 'lichess']
        if platform in platforms_needing_url and not profile_url:
            flash(f'{platform.title()} requires a profile URL', 'error')
            return redirect(url_for('add_app'))
        
        # Validate URL format
        if profile_url and not profile_url.startswith(('http://', 'https://')):
            profile_url = f'https://{profile_url}'
        
        # Create connection
        connection = AppConnection(
            user_id=current_user.id,
            app_name=app_name,
            platform=platform,
            profile_url=profile_url if profile_url else None,
            credentials='{}',
            sync_status='active'
        )
        
        db.session.add(connection)
        db.session.commit()
        
        # Test connection
        if profile_url:
            connector_class = CONNECTOR_REGISTRY.get(platform)
            if connector_class:
                connector = connector_class(connection)
                import asyncio
                is_accessible = asyncio.run(connector.test_connection())
                
                if not is_accessible:
                    connection.sync_status = 'error'
                    db.session.commit()
                    flash(f'Added "{app_name}" but could not access the URL. Check if it\'s correct.', 'warning')
                else:
                    flash(f'App "{app_name}" added successfully!', 'success')
            else:
                flash(f'App "{app_name}" added!', 'success')
        else:
            flash(f'App "{app_name}" added!', 'success')
        
        return redirect(url_for('dashboard'))
    
    # GET request
    platforms = [p.value for p in Platform]
    return render_template('add_app.html', platforms=platforms)


@app.route('/sync/<int:connection_id>')
@login_required
def sync_connection(connection_id):
    """Manually trigger sync for a connection"""
    connection = AppConnection.query.get_or_404(connection_id)
    
    if connection.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    # Trigger async sync
    asyncio.run(dashboard_manager.sync_connection(connection))
    
    return jsonify({'status': 'syncing', 'message': f'Syncing {connection.app_name}...'})


@app.route('/api/app/<int:connection_id>/metrics')
@login_required
def get_app_metrics(connection_id):
    """Get metrics for a specific app"""
    connection = AppConnection.query.get_or_404(connection_id)
    
    if connection.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    metrics = Metric.query.filter_by(connection_id=connection.id)\
        .order_by(Metric.timestamp.desc()).limit(10).all()
    
    return jsonify([{
        'type': m.metric_type,
        'value': m.value,
        'label': m.label,
        'timestamp': m.timestamp.isoformat() if m.timestamp else None
    } for m in metrics])


@app.route('/notifications')
@login_required
def get_notifications():
    """Get notifications for current user"""
    notifications = Notification.query.filter_by(
        user_id=current_user.id
    ).order_by(Notification.created_at.desc()).limit(50).all()
    
    return jsonify([{
        'id': n.id,
        'title': n.title,
        'content': n.content,
        'source': n.source,
        'priority': n.priority,
        'read': n.read,
        'created_at': n.created_at.isoformat() if n.created_at else None
    } for n in notifications])


@app.route('/insights/daily')
@login_required
def daily_insights():
    """Get daily insights"""
    insights = get_daily_insights()
    return jsonify(insights)


def get_daily_insights():
    """Generate or retrieve daily insights"""
    summarizer = NotificationSummarizer()
    
    insights = {
        'health': summarizer.generate_daily_insight('health', ''),
        'law': summarizer.generate_daily_insight('law', ''),
        'philosophy': summarizer.generate_daily_insight('philosophy', ''),
        'bible': summarizer.generate_daily_insight('bible', '')
    }
    
    return insights


# Authentication routes
@app.route('/login')
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('auth.html', mode='login')


@app.route('/login/google')
def google_login():
    """Initiate Google OAuth (mocked for local)"""
    # Mock authentication for local development
    user = User.query.filter_by(email='demo@local.dev').first()
    if not user:
        user = User(email='demo@local.dev', google_id='demo_local')
        db.session.add(user)
        db.session.commit()
    
    login_user(user, remember=True)
    flash('Logged in successfully!', 'success')
    return redirect(url_for('dashboard'))


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!', 'info')
    return redirect(url_for('login'))


# API Routes
@app.route('/api/connections')
@login_required
def api_connections():
    """Get all connections for current user"""
    connections = AppConnection.query.filter_by(user_id=current_user.id).all()
    
    return jsonify([{
        'id': c.id,
        'app_name': c.app_name,
        'platform': c.platform,
        'last_sync': c.last_sync.isoformat() if c.last_sync else None,
        'sync_status': c.sync_status
    } for c in connections])


@app.route('/api/ollama-status')
@login_required
def check_ollama_status():
    """Check if Ollama is running"""
    try:
        summarizer = NotificationSummarizer()
        return jsonify({
            'status': 'connected' if summarizer.ollama_available else 'disconnected',
            'message': 'Ollama is connected and ready for AI insights' if summarizer.ollama_available 
                      else 'Ollama is not running. Using fallback insights.',
            'recommendation': 'Install and run Ollama for AI-powered insights: https://ollama.com/download'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })


@app.route('/api/connection/<int:connection_id>/test')
@login_required
def test_connection(connection_id):
    """Test if a public profile URL is accessible"""
    connection = AppConnection.query.get_or_404(connection_id)
    
    if connection.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    if not connection.profile_url:
        return jsonify({'status': 'error', 'message': 'No profile URL configured'})
    
    connector_class = CONNECTOR_REGISTRY.get(connection.platform)
    if connector_class:
        connector = connector_class(connection)
        import asyncio
        
        try:
            is_accessible = asyncio.run(connector.test_connection())
            if is_accessible:
                return jsonify({
                    'status': 'success',
                    'message': f'Successfully accessed {connection.profile_url}'
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': f'Could not access {connection.profile_url}'
                })
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': str(e)
            })
    
    return jsonify({'status': 'error', 'message': 'Platform not supported'})

@app.route('/auth', methods=['GET', 'POST'], endpoint='login_post')
def login_post():
    """Handle login/registration POST requests"""
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'login':
            # Handle login
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            remember = 'remember' in request.form
            
            user = User.query.filter_by(email=email).first()
            
            if user and user.check_password(password):
                login_user(user, remember=remember)
                flash('Logged in successfully!', 'success')
                
                # Log login activity
                activity = Activity(
                    user_id=user.id,
                    action='login',
                    details='User logged in'
                )
                db.session.add(activity)
                db.session.commit()
                
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password', 'error')
                return redirect(url_for('login'))
        
        elif action == 'register':
            # Handle registration
            name = request.form.get('name', '').strip()
            email = request.form.get('email', '').strip().lower()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirm_password', '')
            
            # Validation
            if not name or not email or not password:
                flash('All fields are required', 'error')
                return redirect(url_for('login'))
            
            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return redirect(url_for('login'))
            
            if len(password) < 8:
                flash('Password must be at least 8 characters', 'error')
                return redirect(url_for('login'))
            
            if not re.match(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)', password):
                flash('Password must include uppercase, lowercase, and numbers', 'error')
                return redirect(url_for('login'))
            
            # Check if user exists
            existing_user = User.query.filter_by(email=email).first()
            if existing_user:
                flash('Email already registered', 'error')
                return redirect(url_for('login'))
            
            # Create new user
            try:
                new_user = User(
                    name=name,
                    email=email,
                    created_at=datetime.utcnow()
                )
                new_user.set_password(password)
                
                db.session.add(new_user)
                db.session.commit()
                
                # Log registration activity
                activity = Activity(
                    user_id=new_user.id,
                    action='register',
                    details='New user registered'
                )
                db.session.add(activity)
                db.session.commit()
                
                # Log user in automatically
                login_user(new_user, remember=True)
                
                flash('Registration successful! Welcome to your dashboard.', 'success')
                
                # In your registration form, add a checkbox:
                # <input type="checkbox" name="create_demo" id="create_demo" checked>
                # <label for="create_demo">Add sample apps to get started</label>

                # Then in your route:
                create_demo = 'create_demo' in request.form
                create_demo_apps_for_user(new_user.id, create_demo=create_demo)
                
                return redirect(url_for('dashboard'))
                
            except Exception as e:
                db.session.rollback()
                flash('Registration failed. Please try again.', 'error')
                logger.error(f"Registration error: {str(e)}")
                return redirect(url_for('login'))
        
        else:
            flash('Invalid action', 'error')
            return redirect(url_for('login'))
    
    # For GET requests, redirect to login page
    return redirect(url_for('login'))

def create_demo_apps_for_user(user_id, create_demo=True):
    """Create demo app connections for new users - OPTIONAL"""
    if not create_demo:
        return
    
    demo_apps = [
        {
            'name': 'GitHub Demo',
            'platform': 'github',
            'sync_status': 'active',
            'profile_url': 'https://github.com/Earl-cmyk'
        },
        {
            'name': 'YouTube Stats',
            'platform': 'youtube',
            'sync_status': 'active',
            'profile_url': 'https://www.youtube.com/@YouTube'
        },
        {
            'name': 'Spotify Playlists',
            'platform': 'spotify',
            'sync_status': 'active',
            'profile_url': 'https://open.spotify.com'
        }
    ]
    
    for app_data in demo_apps:
        app = AppConnection(
            user_id=user_id,
            app_name=app_data['name'],
            platform=app_data['platform'],
            profile_url=app_data.get('profile_url'),
            credentials='{}',
            sync_status=app_data['sync_status'],
            last_sync=datetime.utcnow() if app_data['sync_status'] == 'active' else None
        )
        db.session.add(app)
    
    db.session.commit()
    logger.info(f"Created demo apps for user {user_id}")
    
    # OPTIONAL: Create demo metrics (comment this out for clean start)
    # create_demo_metrics_for_user(user_id)
    
@app.route('/api/ollama/test')
@login_required
def test_ollama():
    """Test Ollama connection and models"""
    try:
        # Test direct connection
        models_response = requests.get('http://localhost:11434/api/tags').json()
        models = models_response.get('models', [])
        
        # Test model generation
        summarizer = NotificationSummarizer()
        
        if summarizer.ollama_available:
            test_prompt = "Hello, are you working?"
            test_response = summarizer.ollama_client.generate(
                model=summarizer.model,
                prompt=test_prompt,
                options={'max_tokens': 20}
            )
            
            return jsonify({
                'status': 'success',
                'ollama_available': True,
                'model': summarizer.model,
                'available_models': [m.get('name') for m in models],
                'test_response': test_response.get('response', 'No response'),
                'recommendation': 'Install llama2:7b for better results' if 'embed' in summarizer.model.lower() else None
            })
        else:
            return jsonify({
                'status': 'success',
                'ollama_available': False,
                'available_models': [m.get('name') for m in models],
                'message': 'Ollama not available, using fallback insights'
            })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'ollama_available': False
        })


# ============================================================================
# NEW ROUTES FOR TABS
# ============================================================================

@app.route('/planner')
@login_required
def planner():
    """Weekly/Daily/Monthly planner with notifications"""
    # Get notifications for dragging
    notifications = Notification.query.filter_by(
        user_id=current_user.id,
        read=False
    ).order_by(Notification.created_at.desc()).limit(20).all()

    # Get calendar events
    events = CalendarEvent.query.filter_by(
        user_id=current_user.id
    ).filter(
        CalendarEvent.start_time >= datetime.utcnow() - timedelta(days=30)
    ).order_by(CalendarEvent.start_time).all()

    # Format for FullCalendar
    calendar_events = []
    for event in events:
        calendar_events.append({
            'id': event.id,
            'title': event.title,
            'start': event.start_time.isoformat(),
            'end': event.end_time.isoformat() if event.end_time else None,
            'allDay': event.all_day,
            'extendedProps': {
                'description': event.description,
                'priority': event.priority,
                'type': event.event_type
            }
        })

    # Stats
    today = datetime.utcnow().date()
    week_start = today - timedelta(days=today.weekday())
    week_end = week_start + timedelta(days=6)

    events_today = CalendarEvent.query.filter_by(
        user_id=current_user.id
    ).filter(
        db.func.date(CalendarEvent.start_time) == today
    ).count()

    events_week = CalendarEvent.query.filter_by(
        user_id=current_user.id
    ).filter(
        CalendarEvent.start_time >= week_start,
        CalendarEvent.start_time <= week_end
    ).count()

    high_priority = CalendarEvent.query.filter_by(
        user_id=current_user.id,
        priority='high'
    ).filter(
        CalendarEvent.start_time >= today
    ).count()

    return render_template('planner.html',
                           notifications=notifications,
                           calendar_events=calendar_events,
                           events_today=events_today,
                           events_week=events_week,
                           high_priority=high_priority)


@app.route('/brain-dump')
@login_required
def brain_dump():
    """Brain dump analysis page"""
    # Get pending notifications for action suggestions
    pending_notifications = Notification.query.filter_by(
        user_id=current_user.id,
        read=False
    ).order_by(Notification.created_at.desc()).limit(10).all()

    return render_template('brain_dump.html',
                           pending_notifications=pending_notifications)


@app.route('/knowledge-library')
@login_required
def knowledge_library():
    """Knowledge library page"""
    # Get knowledge items
    knowledge_items = KnowledgeItem.query.order_by(
        KnowledgeItem.category,
        KnowledgeItem.created_at.desc()
    ).all()

    # Format for template
    formatted_items = []
    for item in knowledge_items:
        formatted_items.append({
            'title': item.title,
            'content': item.content,
            'categories': [item.category],
            'tags': json.loads(item.tags) if item.tags else [],
            'date': item.created_at.strftime('%Y-%m-%d'),
            'source': item.source
        })

    # Check Ollama status
    summarizer = NotificationSummarizer()

    return render_template('knowledge_library.html',
                           knowledge_items=formatted_items,
                           ollama_available=summarizer.ollama_available)


# ============================================================================
# API ROUTES FOR NEW FEATURES
# ============================================================================

@app.route('/api/brain-dump/save', methods=['POST'])
@login_required
def save_brain_dump():
    """Save brain dump content"""
    data = request.json
    content = data.get('content', '')

    # Get or create brain dump
    brain_dump = BrainDump.query.filter_by(
        user_id=current_user.id
    ).order_by(BrainDump.created_at.desc()).first()

    if brain_dump:
        brain_dump.content = content
        brain_dump.updated_at = datetime.utcnow()
    else:
        brain_dump = BrainDump(
            user_id=current_user.id,
            content=content
        )
        db.session.add(brain_dump)

    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/api/brain-dump/latest')
@login_required
def get_latest_brain_dump():
    """Get latest brain dump"""
    brain_dump = BrainDump.query.filter_by(
        user_id=current_user.id
    ).order_by(BrainDump.created_at.desc()).first()

    if brain_dump:
        return jsonify({
            'content': brain_dump.content,
            'created_at': brain_dump.created_at.isoformat()
        })
    return jsonify({'content': ''})


@app.route('/api/brain-dump/analyze', methods=['POST'])
@login_required
def analyze_brain_dump():
    """Analyze brain dump content with AI"""
    data = request.json
    content = data.get('content', '')
    focus = data.get('focus', 'all')

    summarizer = NotificationSummarizer()

    # Use AI analysis if available, otherwise use fallback
    if summarizer.ollama_available:
        try:
            analysis = perform_ai_analysis(content, focus, summarizer)
        except:
            analysis = fallback_brain_analysis(content)
    else:
        analysis = fallback_brain_analysis(content)

    # Save analysis
    brain_dump = BrainDump.query.filter_by(
        user_id=current_user.id
    ).order_by(BrainDump.created_at.desc()).first()

    if brain_dump:
        brain_dump.analysis = json.dumps(analysis)
        db.session.commit()

    return jsonify(analysis)


def perform_ai_analysis(content, focus, summarizer):
    """Perform AI analysis using Ollama"""
    prompts = {
        'tasks': f"""Extract tasks and action items from this text. Format as a bullet list.

        Text: {content[:1000]}

        Tasks:""",

        'health': f"""Analyze health-related concerns or questions in this text. Provide helpful advice.

        Text: {content[:1000]}

        Health Analysis:""",

        'legal': f"""Identify any legal issues or questions in this text. Provide basic guidance based on Philippine law.

        Text: {content[:1000]}

        Legal Analysis:""",

        'philosophy': f"""Provide philosophical insights or perspectives on this text.

        Text: {content[:1000]}

        Philosophical Insights:""",

        'science': f"""Analyze any scientific aspects or questions in this text.

        Text: {content[:1000]}

        Scientific Analysis:"""
    }

    analysis = {}

    if focus == 'all' or focus == 'tasks':
        try:
            response = summarizer.ollama_client.generate(
                model=summarizer.model,
                prompt=prompts['tasks'],
                options={'temperature': 0.3, 'max_tokens': 200}
            )
            analysis['tasks'] = response['response'].strip()
        except:
            analysis['tasks'] = extract_tasks_fallback(content)

    if focus == 'all' or focus in ['health', 'legal', 'philosophy', 'science']:
        categories = ['health', 'legal', 'philosophy', 'science'] if focus == 'all' else [focus]

        for category in categories:
            try:
                insight = summarizer.generate_daily_insight(category, content[:500])
                analysis[category] = insight
            except:
                analysis[category] = fallback_analysis_by_category(category)

    return analysis


def fallback_brain_analysis(content):
    """Fallback analysis when AI is not available"""
    return {
        'tasks': extract_tasks_fallback(content),
        'health': "For health concerns, consult with a healthcare professional.",
        'legal': "For legal matters, consult with a qualified attorney in your jurisdiction.",
        'philosophy': "Take time to reflect on your thoughts. Journaling can provide clarity.",
        'science': "Approach problems systematically and look for evidence-based solutions."
    }


def extract_tasks_fallback(text):
    """Simple task extraction fallback"""
    lines = text.split('\n')
    tasks = []

    for line in lines:
        line = line.strip().lower()
        if any(word in line for word in ['need to', 'should', 'must', 'have to', 'todo', 'task:', '- [ ]']):
            tasks.append(line.capitalize())

    return "\n".join([f"• {task}" for task in tasks[:10]]) if tasks else "No specific tasks identified."


def fallback_analysis_by_category(category):
    """Fallback analysis by category"""
    fallbacks = {
        'health': "Remember to stay hydrated, get enough sleep, and maintain a balanced diet.",
        'legal': "Always document important agreements and understand your rights and responsibilities.",
        'philosophy': "\"The only true wisdom is in knowing you know nothing.\" - Socrates",
        'science': "The scientific method: Observe, question, hypothesize, experiment, analyze, conclude."
    }
    return fallbacks.get(category, "Analysis not available.")


@app.route('/api/calendar/event', methods=['POST'])
@login_required
def create_calendar_event():
    """Create a calendar event"""
    data = request.json

    event = CalendarEvent(
        user_id=current_user.id,
        title=data.get('title'),
        description=data.get('extendedProps', {}).get('description'),
        start_time=datetime.fromisoformat(data.get('start').replace('Z', '+00:00')),
        end_time=datetime.fromisoformat(data.get('end').replace('Z', '+00:00')) if data.get('end') else None,
        all_day=data.get('allDay', False),
        priority=data.get('extendedProps', {}).get('priority', 'medium'),
        event_type=data.get('extendedProps', {}).get('type', 'task')
    )

    db.session.add(event)
    db.session.commit()

    return jsonify({'id': event.id, 'status': 'success'})


@app.route('/api/notification/<int:notification_id>/schedule', methods=['POST'])
@login_required
def schedule_notification(notification_id):
    """Schedule a notification as a calendar event"""
    notification = Notification.query.get_or_404(notification_id)

    if notification.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403

    # Create calendar event
    event = CalendarEvent(
        user_id=current_user.id,
        title=notification.title,
        description=notification.content,
        start_time=datetime.utcnow() + timedelta(hours=1),  # Default: 1 hour from now
        all_day=False,
        event_type='notification',
        source_id=notification.id,
        source_type='notification',
        priority='medium'
    )

    db.session.add(event)
    notification.read = True
    db.session.commit()

    return jsonify({'status': 'success', 'event_id': event.id})


@app.route('/api/library/insight/<category>')
@login_required
def get_library_insight(category):
    """Get an insight from the knowledge library"""
    valid_categories = ['health', 'law', 'philosophy', 'science']

    if category not in valid_categories:
        return jsonify({'error': 'Invalid category'}), 400

    # Try to get from knowledge library
    item = KnowledgeItem.query.filter_by(
        category=category
    ).order_by(db.func.random()).first()

    if item:
        insight = f"{item.title}\n\n{item.content[:200]}..."
    else:
        # Fallback
        summarizer = NotificationSummarizer()
        insight = summarizer.generate_daily_insight(category)

    return jsonify({'insight': insight})


# Add this to your initialization function to populate the knowledge library
def init_knowledge_library():
    """Initialize knowledge library with fallback content"""
    with app.app_context():
        if KnowledgeItem.query.count() == 0:
            knowledge_items = [
                # Health items
                KnowledgeItem(
                    category='health',
                    title='The Importance of Sleep',
                    content='Adults need 7-9 hours of sleep per night for optimal health. Sleep helps with memory consolidation, immune function, and cellular repair.',
                    tags='["sleep", "health", "wellness"]',
                    is_fallback=True
                ),

                # Law items
                KnowledgeItem(
                    category='law',
                    title='Basic Rights in the Philippines',
                    content='The 1987 Philippine Constitution guarantees rights including: freedom of speech, freedom of religion, right to due process, and protection against unreasonable searches.',
                    tags='["law", "rights", "philippines"]',
                    is_fallback=True
                ),

                # Philosophy items
                KnowledgeItem(
                    category='philosophy',
                    title='Stoic Principles',
                    content='Stoicism teaches: focus on what you can control, accept what you cannot, and cultivate virtue. As Marcus Aurelius said, "You have power over your mind - not outside events."',
                    tags='["philosophy", "stoicism", "wisdom"]',
                    is_fallback=True
                ),

                # Science items
                KnowledgeItem(
                    category='science',
                    title='Critical Thinking',
                    content='The scientific method involves: observation, questioning, hypothesis formation, experimentation, analysis, and conclusion. Always question sources and look for evidence.',
                    tags='["science", "thinking", "method"]',
                    is_fallback=True
                ),
            ]

            for item in knowledge_items:
                db.session.add(item)

            db.session.commit()
            logger.info("Initialized knowledge library with fallback content")

@app.route('/api/add-sample-apps', methods=['POST'])
@login_required
def add_sample_apps():
    """Manually add sample apps for exploration"""
    sample_apps = [
        {
            'name': 'GitHub Sample',
            'platform': 'github',
            'profile_url': 'https://github.com/Earl-cmyk'
        },
        {
            'name': 'YouTube Sample',
            'platform': 'youtube',
            'profile_url': 'https://www.youtube.com/@YouTube'
        }
    ]
    
    added_count = 0
    for app_data in sample_apps:
        # Check if already exists
        existing = AppConnection.query.filter_by(
            user_id=current_user.id,
            app_name=app_data['name']
        ).first()
        
        if not existing:
            app = AppConnection(
                user_id=current_user.id,
                app_name=app_data['name'],
                platform=app_data['platform'],
                profile_url=app_data.get('profile_url'),
                credentials='{}',
                sync_status='active',
                last_sync=datetime.utcnow()
            )
            db.session.add(app)
            added_count += 1
    
    if added_count > 0:
        db.session.commit()
        return jsonify({
            'status': 'success',
            'message': f'Added {added_count} sample apps',
            'count': added_count
        })
    else:
        return jsonify({
            'status': 'info',
            'message': 'Sample apps already added'
        })

@app.route('/api/reset-my-apps', methods=['POST'])
@login_required
def reset_my_apps():
    """Remove all apps and metrics for current user"""
    if not request.json or not request.json.get('confirm'):
        return jsonify({'error': 'Confirmation required'}), 400
    
    try:
        # Delete all user's connections
        connections = AppConnection.query.filter_by(user_id=current_user.id).all()
        connection_ids = [c.id for c in connections]
        
        # Delete metrics for these connections
        Metric.query.filter(Metric.connection_id.in_(connection_ids)).delete()
        
        # Delete connections
        for conn in connections:
            db.session.delete(conn)
        
        # Delete brain dumps
        BrainDump.query.filter_by(user_id=current_user.id).delete()
        
        # Delete calendar events
        CalendarEvent.query.filter_by(user_id=current_user.id).delete()
        
        db.session.commit()
        
        # Log activity
        activity = Activity(
            user_id=current_user.id,
            action='reset_apps',
            details='User reset all apps and data'
        )
        db.session.add(activity)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'message': 'All apps and data removed. Start fresh!'
        })
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500  
# ============================================================================
# INITIALIZATION
# ============================================================================

def init_db():
    """Initialize database with all tables"""
    with app.app_context():
        # Create all tables
        db.create_all()

        # Create demo user if none exists
        if not User.query.first():
            demo_user = User(email='demo@local.dev', google_id='demo_local')
            demo_user.set_password('demo123')  # Add a password
            db.session.add(demo_user)
            db.session.commit()

            logger.info("Created demo user")

            # Add some demo connections
            demo_connections = [
                {
                    'app_name': 'GitHub',
                    'platform': 'github',
                    'profile_url': 'https://github.com/Earl-cmyk',
                    'sync_status': 'active'
                },
                {
                    'app_name': 'YouTube',
                    'platform': 'youtube',
                    'profile_url': 'https://www.youtube.com/@YouTube',
                    'sync_status': 'active'
                },
                {
                    'app_name': 'MyFitnessPal',
                    'platform': 'myfitnesspal',
                    'profile_url': None,
                    'sync_status': 'active'
                }
            ]

            for d in demo_connections:
                conn = AppConnection(
                    user_id=demo_user.id,
                    app_name=d['app_name'],
                    platform=d['platform'],
                    profile_url=d.get('profile_url'),
                    credentials='{}',
                    sync_status=d.get('sync_status', 'active'),
                    last_sync=datetime.utcnow() if d.get('sync_status') == 'active' else None
                )
                db.session.add(conn)

            db.session.commit()
            logger.info("Created demo connections")



def init_knowledge_library():
    """Initialize knowledge library with fallback content"""
    try:
        # First, check if the table exists by trying to query it
        try:
            # This will fail if table doesn't exist
            count = KnowledgeItem.query.count()
        except Exception as e:
            logger.warning(f"KnowledgeItem table doesn't exist yet: {e}")
            return

        # Only initialize if table is empty
        if count == 0:
            knowledge_items = [
                # Health items
                KnowledgeItem(
                    category='health',
                    title='The Importance of Sleep',
                    content='Adults need 7-9 hours of sleep per night for optimal health. Sleep helps with memory consolidation, immune function, and cellular repair.',
                    tags=json.dumps(["sleep", "health", "wellness"]),
                    is_fallback=True
                ),

                # Law items
                KnowledgeItem(
                    category='law',
                    title='Basic Rights in the Philippines',
                    content='The 1987 Philippine Constitution guarantees rights including: freedom of speech, freedom of religion, right to due process, and protection against unreasonable searches.',
                    tags=json.dumps(["law", "rights", "philippines"]),
                    is_fallback=True
                ),

                # Philosophy items
                KnowledgeItem(
                    category='philosophy',
                    title='Stoic Principles',
                    content='Stoicism teaches: focus on what you can control, accept what you cannot, and cultivate virtue. As Marcus Aurelius said, "You have power over your mind - not outside events."',
                    tags=json.dumps(["philosophy", "stoicism", "wisdom"]),
                    is_fallback=True
                ),

                # Science items
                KnowledgeItem(
                    category='science',
                    title='Critical Thinking',
                    content='The scientific method involves: observation, questioning, hypothesis formation, experimentation, analysis, and conclusion. Always question sources and look for evidence.',
                    tags=json.dumps(["science", "thinking", "method"]),
                    is_fallback=True
                ),
            ]

            for item in knowledge_items:
                db.session.add(item)

            db.session.commit()
            logger.info("Initialized knowledge library with fallback content")

    except Exception as e:
        logger.error(f"Failed to initialize knowledge library: {e}")
        # Don't crash if this fails


# Initialize dashboard manager
dashboard_manager = None


# ============================================================================
# INITIALIZATION
# ============================================================================

def create_all_tables():
    """Create all database tables"""
    with app.app_context():
        db.create_all()
        logger.info("All database tables created")


def initialize_application():
    """Initialize the entire application"""
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)

    # Create all tables first
    create_all_tables()

    # Initialize data
    with app.app_context():
        # Check if we need to initialize data
        if not User.query.first():
            # Create demo user
            demo_user = User(email='demo@local.dev', google_id='demo_local')
            demo_user.set_password('demo123')
            db.session.add(demo_user)
            db.session.commit()

            logger.info("Created demo user")

            # Create demo connections
            demo_connections = [
                {
                    'app_name': 'GitHub',
                    'platform': 'github',
                    'profile_url': 'https://github.com/Earl-cmyk',
                    'sync_status': 'active'
                },
                {
                    'app_name': 'YouTube',
                    'platform': 'youtube',
                    'profile_url': 'https://www.youtube.com/@YouTube',
                    'sync_status': 'active'
                },
            ]

            for d in demo_connections:
                conn = AppConnection(
                    user_id=demo_user.id,
                    app_name=d['app_name'],
                    platform=d['platform'],
                    profile_url=d.get('profile_url'),
                    credentials='{}',
                    sync_status='active',
                    last_sync=datetime.utcnow()
                )
                db.session.add(conn)

            db.session.commit()

            # Add knowledge library items
            try:
                # Add some knowledge items
                knowledge_items = [
                    KnowledgeItem(
                        category='health',
                        title='Sleep Importance',
                        content='Get 7-9 hours of sleep nightly for optimal health.',
                        tags=json.dumps(["sleep", "health"]),
                        is_fallback=True
                    ),
                    KnowledgeItem(
                        category='law',
                        title='Philippine Rights',
                        content='The Constitution guarantees freedom of speech and religion.',
                        tags=json.dumps(["law", "rights"]),
                        is_fallback=True
                    ),
                ]

                for item in knowledge_items:
                    db.session.add(item)

                db.session.commit()
                logger.info("Added knowledge library items")
            except Exception as e:
                logger.warning(f"Could not add knowledge items: {e}")

    logger.info("Application initialization complete")

def init_knowledge_library_fallback():
    """Initialize knowledge library with fallback content (no user required)"""
    try:
        with app.app_context():
            # Check if knowledge library is empty
            if KnowledgeItem.query.count() == 0:
                knowledge_items = [
                    KnowledgeItem(
                        category='health',
                        title='The Importance of Sleep',
                        content='Adults need 7-9 hours of sleep per night for optimal health. Sleep helps with memory consolidation, immune function, and cellular repair.',
                        tags=json.dumps(["sleep", "health", "wellness"]),
                        is_fallback=True
                    ),
                    KnowledgeItem(
                        category='law',
                        title='Basic Rights in the Philippines',
                        content='The 1987 Philippine Constitution guarantees rights including: freedom of speech, freedom of religion, right to due process, and protection against unreasonable searches.',
                        tags=json.dumps(["law", "rights", "philippines"]),
                        is_fallback=True
                    ),
                    KnowledgeItem(
                        category='philosophy',
                        title='Stoic Principles',
                        content='Stoicism teaches: focus on what you can control, accept what you cannot, and cultivate virtue. As Marcus Aurelius said, "You have power over your mind - not outside events."',
                        tags=json.dumps(["philosophy", "stoicism", "wisdom"]),
                        is_fallback=True
                    ),
                    KnowledgeItem(
                        category='science',
                        title='Critical Thinking',
                        content='The scientific method involves: observation, questioning, hypothesis formation, experimentation, analysis, and conclusion. Always question sources and look for evidence.',
                        tags=json.dumps(["science", "thinking", "method"]),
                        is_fallback=True
                    ),
                ]
                
                for item in knowledge_items:
                    db.session.add(item)
                
                db.session.commit()
                logger.info("Knowledge library initialized")
                
    except Exception as e:
        logger.error(f"Failed to initialize knowledge library: {e}")

def init_db():
    """Initialize database with all tables - NO DEMO DATA"""
    with app.app_context():
        # Create all tables
        db.create_all()
        logger.info("Database tables created")
        
        # COMMENT OUT or REMOVE the demo user creation
        # Users will register themselves
        
        # Optional: You can keep the knowledge library initialization
        init_knowledge_library_fallback()
        
if __name__ == '__main__':
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    
    # Initialize with NO DEMO DATA
    with app.app_context():
        # Create all tables
        db.create_all()
        logger.info("Database tables created")
        
        # Initialize knowledge library only (no user data)
        try:
            if KnowledgeItem.query.count() == 0:
                init_knowledge_library_fallback()
        except:
            pass  # Skip if fails
    
    # Initialize dashboard manager
    dashboard_manager = DashboardManager(app)
    dashboard_manager.start()
    
    # Run Flask app
    try:
        logger.info("Starting Personal Dashboard on http://localhost:5000")
        logger.info("Users will start with a blank slate")
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        if dashboard_manager:
            dashboard_manager.stop()
    except Exception as e:
        logger.error(f"Failed to start app: {e}")
        if dashboard_manager:
            dashboard_manager.stop()