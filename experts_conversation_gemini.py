# app.py

import streamlit as st
import os
from groq import Groq
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from collections import Counter, deque
from typing import Dict, List, Set, Tuple, Optional
import google.generativeai as genai


# Constants
MAX_RETRIES = 3
RETRY_DELAY = 2
TOKEN_LIMIT = 7000
API_RATE_LIMIT = 30
API_TIME_WINDOW = 60

# 로깅 설정
def setup_logging():
    """로깅 설정"""
    try:
        # 현재 디렉토리에 logs 폴더 생성
        current_dir = Path(__file__).parent
        log_dir = current_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # 현재 시간으로 로그 파일명 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'conversation_log_{timestamp}.log'
        
        # 기존 핸들러 제거
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        # 로깅 포맷 설정
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()  # 콘솔 출력용
            ]
        )
        
        logging.info("=== 로깅 시작 (UTF-8 인코딩) ===")
        logging.info(f"로그 파일 경로: {log_file}")
        return True
        
    except Exception as e:
        print(f"로깅 설정 중 오류 발생: {str(e)}")
        return False

def log_conversation_state(state_name, content):
    """대화 상태를 로그에 기록"""
    try:
        logging.info(f"\n=== {state_name} ===\n{json.dumps(content, ensure_ascii=False, indent=2)}")
    except Exception as e:
        logging.error(f"로깅 중 오류 발생: {str(e)}")

def load_api_key():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'gemini_api.txt')  # groq_api.txt -> gemini_api.txt
        with open(file_path, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        return None


def initialize_session_state():
    """세션 상태 초기화"""
    try:
        if 'conversation_context' not in st.session_state:
            st.session_state.conversation_context = ConversationContext()
            logging.info("Conversation context initialized")
            
        if 'rate_limiter' not in st.session_state:
            st.session_state.rate_limiter = APIRateLimiter()
            logging.info("Rate limiter initialized")
            
        if 'expert_data' not in st.session_state:
            st.session_state.expert_data = []
            logging.info("Expert data initialized")
            
        if 'current_round' not in st.session_state:
            st.session_state.current_round = 0
            logging.info("Round counter initialized")
            
        if 'session_keywords' not in st.session_state:
            st.session_state.session_keywords = set()
            logging.info("Session keywords initialized")
            
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
            logging.info("Conversation history initialized")
            
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = load_user_profile()
            logging.info("User profile initialized")
        
        logging.info("Session state initialization completed")
        
    except Exception as e:
        error_msg = f"Session state initialization error: {str(e)}"
        logging.error(error_msg)
        raise Exception(error_msg)

def initialize_api_key():
    """Initialize or update API key from user input"""
    st.sidebar.markdown("## 🔑 API Configuration")
    
    # Session state에 저장된 API 키 확인
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None
        
    # API 키 입력 필드 (password 타입으로 마스킹)
    api_key_input = st.sidebar.text_input(
        "Enter Gemini API Key",
        type="password",
        placeholder="Enter your API key here...",
        value=st.session_state.api_key if st.session_state.api_key else "",
        help="Your API key will be stored only for this session"
    )
    
    # API 키 적용 버튼
    if st.sidebar.button("Apply API Key"):
        if api_key_input:
            st.session_state.api_key = api_key_input
            try:
                # Gemini 클라이언트 초기화 테스트
                genai.configure(api_key=api_key_input)
                test_client = genai.GenerativeModel('gemini-1.5-pro-latest')
                st.sidebar.success("✅ API key successfully configured!")
                return True
            except Exception as e:
                st.sidebar.error(f"❌ Invalid API key: {str(e)}")
                st.session_state.api_key = None
                return False
        else:
            st.sidebar.warning("⚠️ Please enter an API key")
            return False
    
    # 이미 유효한 API 키가 있는 경우
    return bool(st.session_state.api_key)


def analyze_previous_logs():
    """이전 대화 로그들을 분석하여 사용자 프로필 생성"""
    try:
        # 로그 파일들 찾기
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_files = [f for f in os.listdir(base_dir) if f.startswith('conversation_log_') and f.endswith('.log')]
        
        if not log_files:
            return None
            
        # 최근 3개의 로그 파일만 분석
        recent_logs = sorted(log_files, reverse=True)[:3]
        user_interactions = []
        
        for log_file in recent_logs:
            with open(os.path.join(base_dir, log_file), 'r', encoding='utf-8') as f:
                content = f.read()
                # 사용자 입력과 대화 패턴 분석
                user_interactions.extend([line for line in content.split('\n') if '사용자 입력' in line])
        
        return analyze_user_profile(user_interactions)
    except Exception as e:
        logging.error(f"로그 분석 중 오류 발생: {str(e)}")
        return None

def analyze_user_profile(interactions):
    """사용자 상호작용을 분석하여 프로필 생성"""
    if not interactions:
        return {
            "expertise_level": "intermediate",  # 기본값
            "interest_areas": [],
            "complexity_preference": 0.7
        }
    
    # 사용자 입력의 복잡도와 전문성 분석
    total_words = 0
    technical_terms = 0
    topics = []
    
    for interaction in interactions:
        try:
            content = json.loads(interaction.split("사용자 입력 ===")[-1].strip())
            user_input = content.get("user_todo", "")
            
            words = user_input.split()
            total_words += len(words)
            
            # 기술 용어나 전문 용어 카운트 (간단한 예시)
            technical_terms += sum(1 for word in words if len(word) > 7)
            
            # 주제 영역 추출
            topics.extend([word for word in words if len(word) > 3])
        except:
            continue
    
    # 전문성 수준 계산
    if total_words > 0:
        expertise_ratio = technical_terms / total_words
        if expertise_ratio > 0.3:
            level = "expert"
        elif expertise_ratio > 0.1:
            level = "intermediate"
        else:
            level = "beginner"
    else:
        level = "intermediate"
    
    return {
        "expertise_level": level,
        "interest_areas": list(set(topics))[:5],  # 상위 5개 관심 영역
        "complexity_preference": min(0.9, 0.5 + expertise_ratio)
    }

def summarize_conversation(client: genai.GenerativeModel, conversation_history: List[Dict]) -> str:
    """전체 대화 내용을 요약"""
    try:
        # 대화 내용을 구조화된 형태로 변환
        conversation_text = ""
        for entry in conversation_history:
            conversation_text += f"\nExpert ({entry['expert']}): {entry['expert_response']}\n"
            if 'manager_feedback' in entry:
                conversation_text += f"Manager Feedback: {entry['manager_feedback']}\n"
            if 'keywords' in entry:
                conversation_text += f"Keywords: {', '.join(entry['keywords'])}\n"
        
        summary_prompt = f"""
        Please provide a comprehensive summary of the following expert discussion.
        Focus on:
        1. Main topics and key insights
        2. Different perspectives from experts
        3. Critical conclusions
        4. Areas of agreement and disagreement
        5. Key recommendations or action items

        Discussion Content:
        {conversation_text}

        Please structure the summary as follows:
        🎯 Main Topics:
        - [Key topics discussed]

        💡 Key Insights:
        - [Major insights and findings]

        👥 Expert Perspectives:
        - [Different viewpoints]

        ✨ Conclusions:
        - [Main conclusions]

        📋 Recommendations:
        - [Action items or suggestions]
        """

        # Gemini 생성 설정
        generation_config = {
            'temperature': 0.3,  # 더 사실적인 요약을 위해 낮은 temperature
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 2048,
        }

        # 요약 생성
        response = client.generate_content(
            summary_prompt,
            generation_config=generation_config
        )

        summary = response.text.strip()
        logging.info(f"Conversation summary generated successfully")
        return summary

    except Exception as e:
        error_msg = f"Error generating conversation summary: {str(e)}"
        logging.error(error_msg)
        return f"Failed to generate summary: {str(e)}"

def save_conversation_summary(summary: str, user_todo: str) -> str:
    """Save conversation summary to a file"""
    try:
        # 현재 디렉토리에 summaries 폴더 생성
        current_dir = Path(__file__).parent
        summary_dir = current_dir / 'summaries'
        summary_dir.mkdir(exist_ok=True)
        
        # 현재 시간으로 파일명 생성
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 주제를 파일명에 포함 (특수문자 제거 및 길이 제한)
        topic = "".join(c for c in user_todo[:30] if c.isalnum() or c.isspace()).strip()
        topic = topic.replace(' ', '_')
        
        file_name = f"summary_{timestamp}_{topic}.md"
        file_path = summary_dir / file_name
        
        # 마크다운 형식으로 저장
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(f"# Discussion Summary\n\n")
            f.write(f"**Topic**: {user_todo}\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary Content\n\n")
            f.write(summary)
            
        logging.info(f"Summary saved to: {file_path}")
        return str(file_path)
        
    except Exception as e:
        error_msg = f"Error saving summary: {str(e)}"
        logging.error(error_msg)
        raise

def create_expert_team(client: genai.GenerativeModel, user_todo, user_profile=None, max_retries=3):
    """전문가 팀 구성 함수"""
    for attempt in range(max_retries):
        try:
            expert_team_prompt = create_expert_team_prompt(user_todo, user_profile)
            
            # 로깅 추가
            logging.info(f"전문가 팀 생성 프롬프트 (시도 {attempt + 1}):\n{expert_team_prompt}")
            
            # Gemini 생성 설정
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
                'candidate_count': 1,
            }
            
            # Gemini API 호출
            response = client.generate_content(
                expert_team_prompt,
                generation_config=generation_config
            )
            
            # 응답 텍스트 추출
            response_text = response.text.strip()
            
            # 응답 로깅
            logging.info(f"LLM 응답 (시도 {attempt + 1}):\n{response_text}")
            
            # JSON 부분만 추출
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logging.warning(f"JSON 형식을 찾을 수 없음 (시도 {attempt + 1})")
                continue
                
            json_content = response_text[json_start:json_end]
            
            # JSON 파싱 시도
            try:
                experts_config = json.loads(json_content)
            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 오류 (시도 {attempt + 1}): {str(e)}\n응답: {json_content}")
                continue
            
            if 'experts' not in experts_config:
                logging.warning(f"experts 키가 없음 (시도 {attempt + 1})")
                continue
            
            # 전문가 데이터 검증
            experts = experts_config['experts']
            all_valid = True
            error_message = ""
            
            for expert in experts:
                is_valid, error = validate_expert_data(expert)
                if not is_valid:
                    all_valid = False
                    error_message = error
                    logging.error(f"전문가 데이터 검증 실패: {error}")
                    break
            
            if all_valid:
                logging.info("전문가 팀 생성 성공")
                return True, experts
            else:
                if attempt == max_retries - 1:
                    return False, f"전문가 데이터 검증 실패: {error_message}"
                continue
                
        except Exception as e:
            logging.error(f"예상치 못한 오류 (시도 {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                return False, f"예상치 못한 오류가 발생했습니다: {str(e)}"
            continue
    
    return False, "최대 재시도 횟수를 초과했습니다."

def format_conversation_history(history, max_entries=6):
    """Format recent conversation history"""
    if not history:
        return ""
    
    recent_history = history[-max_entries:]
    formatted = "Previous Discussion:\n"
    
    for entry in recent_history:
        expert_name = entry.get('expert', 'Unknown Expert')
        expert_response = entry.get('expert_response', '')
        manager_feedback = entry.get('manager_feedback', '')
        
        formatted += f"Expert ({expert_name}): {expert_response}\n"
        if manager_feedback:
            formatted += f"Manager Assessment: {manager_feedback}\n"
    
    return formatted

def create_expert_prompt(expert, conversation_history):
    """Create prompt for expert response"""
    return f"""
    You are an expert with the following characteristics:
    - Name: {expert['name']} (Age: {expert['age']}, Gender: {expert['gender']})
    - Nationality: {expert['nationality']}
    - Expertise: {expert['expertise']}
    - Personality: {expert['personality']}
    - Communication Style: {expert['speaking_style']}
    
    Interaction Parameters:
    - Positivity: {expert['positivity']}
    - Verbosity: {expert['verbosity']}
    - Explanation Enthusiasm: {expert['explanation_enthusiasm']}

    {format_conversation_history(conversation_history)}

    Consider the opinions of other experts and manager feedback.
    Provide your expert opinion within 200 characters, leveraging your expertise and characteristics.
    """

class ConversationContext:
    def __init__(self, max_history: int = 5):
        self.history: deque = deque(maxlen=max_history)
        self.keywords: Set[str] = set()
        self.expert_insights: Dict[str, List[str]] = {}
        self.current_topic: str = ""
        self.discussion_depth: int = 0
        self.last_update: float = time.time()

    def add_response(self, expert_name: str, expert_response: str, keywords: List[str]) -> None:
        """Add a response to the conversation history with validation"""
        try:
            if not all([expert_name, expert_response]):
                raise ValueError("Expert name and response are required")
                
            entry = {
                "expert": expert_name,
                "expert_response": expert_response,  # 키 이름 변경
                "keywords": keywords,
                "timestamp": time.time()
            }
            self.history.append(entry)
            self.keywords.update(keywords)
            
            if expert_name not in self.expert_insights:
                self.expert_insights[expert_name] = []
            self.expert_insights[expert_name].append(expert_response)
            
            self.discussion_depth += 1
            self.last_update = time.time()
            
            logging.info(f"Added response from {expert_name} with {len(keywords)} keywords")
            
        except Exception as e:
            logging.error(f"Error adding response: {str(e)}")
            raise


class APIRateLimiter:
    def __init__(self, max_requests: int = API_RATE_LIMIT, time_window: int = API_TIME_WINDOW):
        self.requests: deque = deque()
        self.max_requests = max_requests
        self.time_window = time_window
        self.last_retry_time: float = 0
    
    def can_make_request(self) -> bool:
        """Check if a request can be made with cleanup"""
        current_time = time.time()
        
        # Clean up old requests
        while self.requests and current_time - self.requests[0] > self.time_window:
            self.requests.popleft()
            
        # Check if we're within rate limits
        return len(self.requests) < self.max_requests
    
    def add_request(self) -> None:
        """Record a new request"""
        self.requests.append(time.time())
        
    def wait_if_needed(self) -> None:
        """Wait if rate limit is reached"""
        if not self.can_make_request():
            wait_time = self.time_window - (time.time() - self.requests[0])
            if wait_time > 0:
                logging.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds")
                time.sleep(wait_time)

def extract_keywords(text: str, client: genai.GenerativeModel) -> List[str]:
    """Extract keywords using Gemini"""
    try:
        keyword_prompt = f"""
        Extract the most important technical or domain-specific keywords from the following text.
        Return only the keywords as a comma-separated list, without explanations or additional text.
        Focus on meaningful terms and concepts, avoiding common words.
        
        Text: {text}
        
        Keywords:"""
        
        response = client.generate_content(keyword_prompt)
        
        # 응답에서 키워드 추출 및 정리
        keywords_text = response.text.strip()
        keywords = [k.strip() for k in keywords_text.split(',') if k.strip()]
        
        logging.info(f"Extracted {len(keywords)} keywords: {', '.join(keywords)}")
        return keywords
        
    except Exception as e:
        logging.error(f"Error in keyword extraction: {str(e)}")
        return []

def get_expert_response(
    client: genai.GenerativeModel,  # 타입 힌트 변경
    expert: Dict,
    user_todo: str,
    context: ConversationContext,
    rate_limiter: APIRateLimiter
) -> Tuple[str, List[str]]:
    """Get expert response with rate limiting and retries"""
    try:
        rate_limiter.wait_if_needed()
        
        # Build the system prompt with expert's characteristics
        system_prompt = f"""You are {expert['name']}, an expert with the following characteristics:
- Role: {expert['role']}
- Expertise: {expert['expertise']}
- Nationality: {expert['nationality']}
- Personality: {expert['personality']}
- Speaking Style: {expert['speaking_style']}
- Cultural Perspective: {expert['cultural_bias']}

Please analyze the given task from your unique perspective and expertise.
"""

        # Combine system and user prompts for Gemini
        combined_prompt = f"{system_prompt}\n\n{user_todo}"
        
        generation_config = {
            'temperature': expert.get('temperature', 0.7),
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 2048,
            'candidate_count': 1,
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                rate_limiter.add_request()
                response = client.generate_content(
                    combined_prompt,
                    generation_config=generation_config
                )
                
                expert_response = response.text.strip()
                # Extract keywords using the same model
                keywords = extract_keywords(expert_response, client)
                
                logging.info(f"Expert {expert['name']} response generated successfully")
                return expert_response, keywords
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                time.sleep(RETRY_DELAY * (attempt + 1))
                
    except Exception as e:
        logging.error(f"Error in expert response: {str(e)}")
        raise


def format_previous_insights(context: ConversationContext) -> str:
    """Format previous expert insights for context"""
    insights = []
    for entry in list(context.history)[-3:]:  # Include the last 3 responses
        insights.append(f"{entry['expert']}: {entry['expert_response']}")
    return "\n\n".join(insights) if insights else "No previous insights available."

def format_manager_feedback(context: ConversationContext) -> str:
    """Format manager feedback for context"""
    feedbacks = []
    for entry in list(context.history)[-3:]:
        if entry.get('manager_feedback'):
            feedbacks.append(f"Feedback on {entry['expert']}:\n{entry['manager_feedback']}")
    return "\n\n".join(feedbacks) if feedbacks else "No manager feedback available."


def get_manager_response(client: genai.GenerativeModel, expert: Dict, expert_response: str, context: ConversationContext):
    """Enhanced manager evaluation with context awareness"""
    try:
        manager_eval_prompt = f"""
        Evaluate {expert['name']}'s contribution in the context of the ongoing discussion:
        
        Discussion Progress: Round {context.discussion_depth}/5
        Current Keywords: {', '.join(sorted(context.keywords))}
        
        Latest Response: {expert_response}
        
        Previous Insights:
        {format_previous_insights(context)}
        
        Provide evaluation in this format:
        ✓ New Insights:
        - Focus on unique contributions
        
        ⚠️ Development Needs:
        - Identify gaps in analysis
        
        ➜ Discussion Direction:
        - Suggest next focus areas
        
        💡 Cross-Expert Prompts:
        - Specific questions for other experts
        """
        
        # Gemini 생성 설정
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 2048,
            'candidate_count': 1,
        }
        
        # Gemini API 호출
        response = client.generate_content(
            f"""You are a strategic discussion facilitator focused on driving deeper insights.
            
            {manager_eval_prompt}""",
            generation_config=generation_config
        )
        
        return response.text.strip()
        
    except Exception as e:
        logging.error(f"Manager evaluation error: {str(e)}")
        raise

def validate_expert_data(expert):
    """전문가 데이터의 필수 항목을 검증하는 함수"""
    required_fields = {
        'name': str,
        'role': str,
        'expertise': str,
        'age': int,
        'gender': str,
        'nationality': str,
        'personality': str,
        'speaking_style': str,
        'cultural_bias': str,
        'positivity': float,
        'verbosity': float,
        'explanation_enthusiasm': float,
        'temperature': float
    }
    
    for field, field_type in required_fields.items():
        if field not in expert:
            return False, f"필수 항목 '{field}'가 누락되었습니다."
        if not isinstance(expert[field], field_type):
            return False, f"'{field}' 항목의 데이터 타입이 잘못되습니다."
    return True, ""

def create_expert_team_prompt(user_todo: str, user_profile: Optional[Dict] = None) -> str:
    """전문가 팀 구성을 위한 프롬프트 생성"""
    
    # 기본 프로필 설정
    default_profile = {
        "expertise_level": "intermediate",
        "interest_areas": [],
        "complexity_preference": 0.7
    }
    
    profile = user_profile if user_profile else default_profile
    
    prompt = f"""
    주어진 작업과 사용자 프로필을 바탕으로 최적의 전문가 팀을 구성해주세요.
    
    작업 내용: {user_todo}
    
    사용자 프로필:
    - 전문성 수준: {profile['expertise_level']}
    - 관심 분야: {', '.join(profile['interest_areas']) if profile['interest_areas'] else '미지정'}
    - 선호 복잡도: {profile['complexity_preference']}
    
    다음 형식의 JSON으로 응답해주세요:
    {{
        "experts": [
            {{
                "name": "전문가 이름",
                "role": "전문가 역할",
                "expertise": "전문 분야",
                "age": 나이(정수),
                "gender": "성별",
                "nationality": "국적",
                "personality": "성격 특성",
                "speaking_style": "대화 스타일",
                "cultural_bias": "문화적 관점",
                "positivity": 긍정성(0.0~1.0),
                "verbosity": 상세도(0.0~1.0),
                "explanation_enthusiasm": 설명열의(0.0~1.0),
                "temperature": 창의성(0.0~1.0)
            }}
        ]
    }}
    
    요구사항:
    1. 각 전문가는 고유한 전문성과 관점을 가져야 합니다
    2. 3-5명의 전문가를 포함해주세요
    3. 다양한 국적과 배경을 고려해주세요
    4. 모든 필드는 필수입니다
    5. 수치형 데이터는 적절한 범위 내에서 지정해주세요
    """
    
    return prompt

def create_expert_team(client: genai.GenerativeModel, user_todo, user_profile=None, max_retries=3):
    """전문가 팀 구성 함수"""
    for attempt in range(max_retries):
        try:
            expert_team_prompt = create_expert_team_prompt(user_todo, user_profile)
            
            # 로깅 추가
            logging.info(f"전문가 팀 생성 프롬프트 (시도 {attempt + 1}):\n{expert_team_prompt}")
            
            # Gemini 생성 설정
            generation_config = {
                'temperature': 0.7,
                'top_p': 0.8,
                'top_k': 40,
                'max_output_tokens': 2048,
            }
            
            # Gemini API 호출 - 시스템 프롬프트와 사용자 프롬프트를 하나의 문자열로 결합
            full_prompt = f"""You are a project manager who responds in precise JSON format.

{expert_team_prompt}"""
            
            response = client.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            # 응답 텍스트 추출
            response_text = response.text.strip()
            
            # 응답 로깅
            logging.info(f"LLM 응답 (시도 {attempt + 1}):\n{response_text}")
            
            # JSON 부분만 추출
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                logging.warning(f"JSON 형식을 찾을 수 없음 (시도 {attempt + 1})")
                continue
                
            json_content = response_text[json_start:json_end]
            
            # JSON 파싱 시도
            try:
                experts_config = json.loads(json_content)
            except json.JSONDecodeError as e:
                logging.error(f"JSON 파싱 오류 (시도 {attempt + 1}): {str(e)}\n응답: {json_content}")
                continue
            
            if 'experts' not in experts_config:
                logging.warning(f"experts 키가 없음 (시도 {attempt + 1})")
                continue
            
            # 전문가 데이터 검증
            experts = experts_config['experts']
            all_valid = True
            error_message = ""
            
            for expert in experts:
                is_valid, error = validate_expert_data(expert)
                if not is_valid:
                    all_valid = False
                    error_message = error
                    logging.error(f"전문가 데이터 검증 실패: {error}")
                    break
            
            if all_valid:
                logging.info("전문가 팀 생성 성공")
                return True, experts
            else:
                if attempt == max_retries - 1:
                    return False, f"전문가 데이터 검증 실패: {error_message}"
                continue
                
        except Exception as e:
            logging.error(f"예상치 못한 오류 (시도 {attempt + 1}): {str(e)}")
            if attempt == max_retries - 1:
                return False, f"예상치 못한 오류가 발생했습니다: {str(e)}"
            continue
    
    return False, "최대 재시도 횟수를 초과했습니다."


def estimate_tokens(text):
    """Estimate token count (roughly 4 chars per token for English)"""
    return len(text) // 4

def check_token_limit(prompt, history, limit=7000):
    """Check token usage"""
    total_tokens = estimate_tokens(prompt)
    for entry in history:
        total_tokens += estimate_tokens(str(entry))
    return total_tokens < limit, total_tokens

def update_user_profile(current_profile, new_keywords, client: genai.GenerativeModel):
    """Update user profile based on keywords"""
    try:
        current_keywords = current_profile.get("keywords", [])
        updated_keywords = list(dict.fromkeys(current_keywords + new_keywords))[-50:]
        
        analysis_prompt = f"""
        Analyze the user profile based on these keywords and their frequencies:
        {json.dumps(dict(Counter(updated_keywords)), ensure_ascii=False)}

        Current Profile:
        {json.dumps(current_profile, ensure_ascii=False)}

        Respond in JSON format:
        {{
            "expertise_level": "beginner/intermediate/expert",
            "interest_areas": ["primary interest 1", "primary interest 2", ...],
            "complexity_preference": 0.1~1.0,
            "progression_note": "brief note about user's growth/changes"
        }}
        """
        
        # Gemini 생성 설정
        generation_config = {
            'temperature': 0.3,
            'top_p': 0.8,
            'top_k': 40,
            'max_output_tokens': 2048,
            'candidate_count': 1,
        }
        
        # Gemini API 호출
        response = client.generate_content(
            f"""You are a user learning pattern analysis expert.
            
            {analysis_prompt}""",
            generation_config=generation_config
        )
        
        updated_analysis = json.loads(response.text)
        
        # 프로필 업데이트
        new_profile = {
            "expertise_level": updated_analysis["expertise_level"],
            "keywords": updated_keywords,
            "interest_areas": updated_analysis["interest_areas"],
            "complexity_preference": updated_analysis["complexity_preference"],
            "last_update": datetime.now().isoformat(),
            "progression_note": updated_analysis.get("progression_note", "")
        }
        
        logging.info(f"프로필 업데이트 완료:\n{json.dumps(new_profile, ensure_ascii=False, indent=2)}")
        return new_profile
        
    except Exception as e:
        logging.error(f"프로필 업데이트 중 오류 발생: {str(e)}")
        return current_profile

def get_profile_directory():
    """프로필 저장 디렉토리 생성 및 반환"""
    # 사용자의 홈 디렉토리 아래에 .experts_llm 디렉토리 생성
    profile_dir = Path.home() / '.experts_llm' / 'profiles'
    profile_dir.mkdir(parents=True, exist_ok=True)
    return profile_dir

def save_user_profile(profile_data: Dict) -> None:
    """Save user profile to JSON file"""
    try:
        # 현재 디렉토리에 profiles 폴더 생성
        current_dir = Path(__file__).parent
        profile_dir = current_dir / 'profiles'
        profile_dir.mkdir(exist_ok=True)
        
        profile_path = profile_dir / 'user_profile.json'
        
        # 기존 프로필이 있다면 읽어오기
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                existing_profile = json.load(f)
        else:
            existing_profile = {
                "expertise_level": "intermediate",
                "keywords": [],
                "interest_areas": [],
                "complexity_preference": 0.7
            }
        
        # 키워드 업데이트
        existing_profile["keywords"] = list(set(existing_profile.get("keywords", []) + profile_data.get("keywords", [])))
        
        # 프로필 저장
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(existing_profile, f, ensure_ascii=False, indent=2)
            
        logging.info(f"User profile saved: {profile_path}")
        
    except Exception as e:
        logging.error(f"Error saving user profile: {str(e)}")



def load_user_profile() -> Dict:
    """Load user profile from JSON file"""
    try:
        current_dir = Path(__file__).parent
        profile_path = current_dir / 'profiles' / 'user_profile.json'
        
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                profile = json.load(f)
                logging.info(f"프로필 로드됨: {profile_path}")
                return profile
    except Exception as e:
        logging.error(f"Error loading user profile: {str(e)}")
    
    # 기본 프로필 반환
    return {
        "expertise_level": "intermediate",
        "keywords": [],
        "interest_areas": [],
        "complexity_preference": 0.7
    }

def show_profile_info():
    """프로필 정보를 사이드바에 표시"""
    try:
        profile_dir = get_profile_directory()
        current_profile = load_user_profile()
        
        st.sidebar.subheader("📊 프로필 정보")
        
        if current_profile:
            st.sidebar.markdown("**현재 프로필:**")
            st.sidebar.json(current_profile)
            
            # 백업 파일 목록
            backup_files = sorted(profile_dir.glob('user_profile_backup_*.json'))
            if backup_files:
                st.sidebar.markdown("**백업 기록:**")
                for backup in backup_files[-3:]:  # 최근 3개만 표시
                    timestamp = backup.stem.split('_')[-1]
                    st.sidebar.text(f"• 백업: {timestamp}")
        else:
            st.sidebar.warning("저장된 프로필이 없습니다.")
        
        st.sidebar.markdown(f"**저장 위치:** `{profile_dir}`")
        
    except Exception as e:
        st.sidebar.error(f"프로필 정보 표시 중 오류 발생: {str(e)}")

def get_custom_css() -> str:
    """Get custom CSS for styling"""
    return """
    <style>
        .expert-response {
            margin: 10px 0;
            padding: 15px;
            border-radius: 5px;
            width: 80%;
            font-size: 0.95em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .manager-response {
            margin-left: 10%;  /* 20% -> 10% 로 변경 */
            padding: 12px;
            border-radius: 5px;
            width: 80%;        /* 20% -> 80% 로 변경 */
            font-size: 0.9em;  /* 0.7em -> 0.9em 으로 변경 */
            background-color: #f5f5f5;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .discussion-progress {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 8px;
            background-color: #f0f8ff;
            border-radius: 4px;
            font-size: 0.8em;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .expert-1 { font-family: 'Arial', sans-serif; }
        .expert-2 { font-family: 'Georgia', serif; }
        .expert-3 { font-family: 'Verdana', sans-serif; }
        .expert-4 { font-family: 'Trebuchet MS', sans-serif; }
        .expert-5 { font-family: 'Palatino', serif; }
        
        .keywords-display {
            margin: 10px 0;
            padding: 10px;
            background-color: #f0f8ff;
            border-radius: 5px;
            font-size: 0.9em;
        }
    </style>
    """

def reset_conversation():
    """대화 상태 초기화"""
    st.session_state.current_round = 0
    st.session_state.conversation_history = []
    st.session_state.expert_data = []
    # 키워드는 유지하되, 현재 세션 키워드만 초기화
    st.session_state.session_keywords = set()

def process_expert_analysis(client: genai.GenerativeModel, user_todo: str) -> None:  # Groq -> GenerativeModel로 변경
    """Process expert analysis with progress tracking and error handling"""
    try:
        with st.spinner('Composing expert team...'):
            success, expert_team = create_expert_team(client, user_todo)
            if not success:
                st.error(f"Failed to create expert team: {expert_team}")
                return
                
            st.session_state.expert_data = expert_team
            logging.info(f"Expert team created with {len(expert_team)} members")
            
        # Create conversation container
        conversation_container = st.container()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Reset session keywords for new conversation
        st.session_state.session_keywords = set()
        st.session_state.conversation_history = []
        
        # Process each expert's response
        for idx, expert in enumerate(expert_team):
            try:
                progress = (idx + 1) / len(expert_team)
                progress_bar.progress(progress)
                status_text.text(f'Processing expert {idx + 1}/{len(expert_team)}...')
                
                # Get expert response
                expert_response, keywords = get_expert_response(
                    client,
                    expert,
                    user_todo,
                    st.session_state.conversation_context,
                    st.session_state.rate_limiter
                )
                
                # Update session keywords and save profile
                if keywords:
                    st.session_state.session_keywords.update(keywords)
                    # 각 전문가 응답 후 프로필 업데이트
                    profile_update = {
                        "keywords": list(st.session_state.session_keywords)
                    }
                    save_user_profile(profile_update)
                    logging.info(f"Profile updated with keywords from {expert['name']}")
                
                # Display expert response
                with conversation_container:
                    st.markdown(f"""
                        <div class="expert-response expert-{idx + 1}">
                            <strong>{expert['name']}</strong>: {expert_response}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Get and display manager response
                    manager_response = get_manager_response(
                        client,
                        expert,
                        expert_response,
                        st.session_state.conversation_context
                    )
                    
                    st.markdown(f"""
                        <div class="manager-response">
                            <strong>Manager</strong>: {manager_response}
                        </div>
                    """, unsafe_allow_html=True)
                
                # Save conversation entry
                conversation_entry = {
                    "expert": expert['name'],
                    "expert_response": expert_response,
                    "manager_feedback": manager_response,
                    "keywords": list(keywords),
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.conversation_history.append(conversation_entry)

                # 대화 컨텍스트 업데이트
                st.session_state.conversation_context.add_response(expert['name'], expert_response, keywords)
                
                # Update sidebar with current keywords and profile info
                with st.sidebar:
                    st.subheader("🔑 Current Session Keywords")
                    st.markdown(f"""
                        <div class="keywords-display">
                            {', '.join(sorted(st.session_state.session_keywords))}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show profile update status
                    st.success(f"Profile updated with {len(keywords)} new keywords")
                
            except Exception as e:
                error_msg = f"Error processing expert {expert['name']}: {str(e)}"
                logging.error(error_msg)
                st.error(error_msg)
                continue
        
        # Complete progress
        progress_bar.progress(1.0)
        status_text.text('Analysis completed! Generating summary...')

        # 대화 요약 생성 및 표시
        with st.spinner('Generating conversation summary...'):
            summary = summarize_conversation(client, st.session_state.conversation_history)
            
            # 요약 표시
            st.markdown("## 💫 Discussion Summary")
            st.markdown(summary)
            
            # 요약을 파일로 저장
            try:
                summary_path = save_conversation_summary(summary, user_todo)
                st.success(f"Summary saved to: {summary_path}")
            except Exception as e:
                st.warning(f"Failed to save summary: {str(e)}")
            
            # 요약을 대화 기록에 저장
            summary_entry = {
                "type": "summary",
                "content": summary,
                "file_path": summary_path,  # 저장된 파일 경로 추가
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.conversation_history.append(summary_entry)
            
            # 로깅
            logging.info("Conversation summary added to history and saved to file")

        status_text.text('Analysis and summary completed!')
        
    except Exception as e:
        error_msg = f"Error in expert analysis: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)



def estimate_token_usage(context: ConversationContext) -> int:
    """Estimate token usage from conversation context"""
    total_tokens = 0
    for entry in context.history:
        # Rough estimate: 1 token per 4 characters
        total_tokens += len(str(entry)) // 4
    return total_tokens

def reset_conversation_state() -> None:
    """Reset conversation state for new analysis"""
    st.session_state.conversation_context = ConversationContext()
    st.session_state.expert_data = []
    st.session_state.session_keywords = set()
    st.session_state.conversation_history = []
    logging.info("Conversation state reset")

def main():
    if not setup_logging():
        st.error("Failed to initialize logging system")
        return
        
    try:
        st.markdown(get_custom_css(), unsafe_allow_html=True)
        initialize_session_state()
        
        st.title("Expert LLM System")
        
        # API 키 초기화 및 검증
        if not initialize_api_key():
            st.warning("Please configure your Gemini API key in the sidebar to continue.")
            return
            
        # Gemini 클라이언트 초기화
        genai.configure(api_key=st.session_state.api_key)
        client = genai.GenerativeModel('gemini-1.5-pro-latest')
        logging.info("API client initialized successfully")
        
        # User input
        user_todo = st.text_area(
            "Enter your task:",
            placeholder="Example: Analyze the impact of AI on healthcare",
            help="Be specific about what you want to analyze"
        )
        
        if st.button("Start Expert Analysis"):
            if not user_todo:
                st.warning("Please enter a task to analyze")
                return
            process_expert_analysis(client, user_todo)
            
    except Exception as e:
        error_msg = f"Application error: {str(e)}"
        logging.error(error_msg)
        st.error(error_msg)

if __name__ == "__main__":
    main()
