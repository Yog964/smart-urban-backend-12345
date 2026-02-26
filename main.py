from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import engine, Base, SessionLocal
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import models, schemas, utils
import os
import uuid
import sys

load_dotenv()

# Add AI-Image-analysis to path so we can import the analyzer
_AI_DIR = os.path.join(os.path.dirname(__file__), "AI-Image-analysis")
if _AI_DIR not in sys.path:
    sys.path.insert(0, _AI_DIR)

try:
    from urban_vision_analyzer import UrbanIssueAnalyzer
    _analyzer = UrbanIssueAnalyzer()
    print("[AI] UrbanIssueAnalyzer loaded ✅")
except Exception as e:
    _analyzer = None
    print(f"[AI] ⚠️ Could not load UrbanIssueAnalyzer: {e}")


# Supabase config from env
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "complaint-images")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

Base.metadata.create_all(bind=engine)

app = FastAPI(title="UrbanSathi Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    from jose import JWTError, jwt
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, utils.SECRET_KEY, algorithms=[utils.ALGORITHM])
        phone: str = payload.get("sub")
        if phone is None:
            raise credentials_exception
        token_data = schemas.TokenData(phone_number=phone)
    except JWTError:
        raise credentials_exception
        
    user = db.query(models.User).filter(models.User.phone_number == token_data.phone_number).first()
    if user is None:
        raise credentials_exception
    return user



@app.post("/register", response_model=schemas.User)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    try:
        print(f"Registering user: {user.phone_number}")
        db_user = db.query(models.User).filter(models.User.phone_number == user.phone_number).first()
        if db_user:
            print("User already exists")
            raise HTTPException(status_code=400, detail="Phone number already registered")
            
        hashed_password = utils.get_password_hash(user.password)
        new_user = models.User(
            phone_number=user.phone_number,
            password=hashed_password,
            name=user.name,
            area=user.area
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        print("User registered successfully")
        return new_user
    except Exception as e:
        print(f"Registration Error: {e}")
        import traceback
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/token", response_model=schemas.Token)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Using 'username' field for 'phone_number' because of OAuth2 spec
    user = db.query(models.User).filter(models.User.phone_number == form_data.username).first()
    if not user or not utils.verify_password(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = utils.create_access_token(data={"sub": user.phone_number})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/users/me", response_model=schemas.User)
def read_users_me(current_user: models.User = Depends(get_current_user)):
    return current_user


# --- Supabase Storage Upload (Backend / service key only) ---
ALLOWED_MIME_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif",
    "audio/mpeg", "audio/mp4", "audio/wav", "audio/ogg", "audio/m4a", "audio/x-m4a",
}
MAX_FILE_SIZE_BYTES = 5 * 1024 * 1024  # 5MB

@app.post("/upload/", response_model=dict)
async def upload_file(file: UploadFile = File(...)):
    print(f"[UPLOAD] Received file: {file.filename}, type: {file.content_type}")

    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise HTTPException(status_code=500, detail="Supabase storage not configured on server.")

    # Validate MIME type
    content_type = file.content_type or "application/octet-stream"
    if content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {content_type}. Allowed: {ALLOWED_MIME_TYPES}")

    file_bytes = await file.read()

    # Validate file size
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=400, detail=f"File too large. Max allowed: 5MB, received: {len(file_bytes) // 1024}KB")

    # Build unique path: public/<timestamp>_<uuid><ext>
    import time
    file_extension = os.path.splitext(file.filename or "file")[1] or ".jpg"
    timestamp = int(time.time() * 1000)
    unique_filename = f"public/{timestamp}_{uuid.uuid4().hex}{file_extension}"

    print(f"[UPLOAD] Uploading to Supabase bucket '{SUPABASE_BUCKET}', path: {unique_filename}")

    import httpx
    upload_url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{unique_filename}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": content_type,
        "x-upsert": "true",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(upload_url, content=file_bytes, headers=headers)
        print(f"[UPLOAD] Supabase response: {res.status_code} — {res.text[:200]}")
        if res.status_code not in (200, 201):
            raise HTTPException(status_code=500, detail=f"Supabase upload failed ({res.status_code}): {res.text}")

    # Build permanent public URL
    public_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{unique_filename}"
    print(f"[UPLOAD] ✅ Success! Public URL: {public_url}")
    return {"image_url": public_url}


# --- AI Analysis endpoint (called by Flutter after image upload) ---
@app.post("/analyze/", response_model=dict)
async def analyze_image_url(payload: dict):
    """
    Accepts { "image_url": "<public supabase url>" }
    Returns { "department", "subcategory", "issue_type", "confidence_percent", "severity_score", "ai_detected" }
    """
    image_url = payload.get("image_url", "")
    print(f"[ANALYZE] Received URL: {image_url}")

    if not image_url or not image_url.startswith("http"):
        return {"ai_detected": False, "department": None, "subcategory": None, "issue_type": "unknown", "confidence_percent": 0, "severity_score": 1}

    if not _analyzer:
        print("[ANALYZE] Analyzer not available")
        return {"ai_detected": False, "department": None, "subcategory": None, "issue_type": "unknown", "confidence_percent": 0, "severity_score": 1}

    result = _analyzer.analyze_image_from_url(image_url)
    print(f"[ANALYZE] Result: {result}")
    return result


@app.post("/complaints/", response_model=schemas.ComplaintAIResponse)
def create_complaint(
    complaint: schemas.ComplaintCreate,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    print(f"[COMPLAINT] Creating for user: {current_user.phone_number}")
    print(f"[COMPLAINT] image_url: {complaint.image_url}")
    print(f"[COMPLAINT] voice_url: {complaint.voice_url}")
    print(f"[COMPLAINT] user-provided dept: {complaint.department}, sub: {complaint.subcategory}")

    # ── AI ANALYSIS ──────────────────────────────────────────────────────────
    ai_department   = None
    ai_subcategory  = None
    ai_severity     = 5.0
    ai_confidence   = 0.0
    ai_issue_type   = "unknown"

    if complaint.image_url and complaint.image_url.startswith("http") and _analyzer:
        print(f"[AI] Running analysis on: {complaint.image_url}")
        try:
            result = _analyzer.analyze_image_from_url(complaint.image_url)
            print(f"[AI] Result → {result}")

            if result.get("ai_detected"):
                ai_issue_type  = result["issue_type"]
                ai_department  = result["department"]
                ai_subcategory = result["subcategory"]
                ai_severity    = float(result.get("severity_score", 5))
                ai_confidence  = float(result.get("confidence_percent", 0))
                print(f"[AI] ✅ Auto-detected: dept={ai_department}, sub={ai_subcategory}, severity={ai_severity}")
            else:
                print(f"[AI] ⚠️ Low confidence or unknown. Falling back to user selection.")
        except Exception as e:
            print(f"[AI] ❌ Analysis failed: {e}")
    else:
        print(f"[AI] Skipped (no image URL or analyzer unavailable)")

    # Fallback: use user-provided values if AI couldn't determine
    final_department  = ai_department  or complaint.department  or "General"
    final_subcategory = ai_subcategory or complaint.subcategory or "Other"
    final_issue_type  = ai_issue_type  if ai_issue_type != "unknown" else (complaint.subcategory or "Other")

    # Auto-generate title if not meaningful
    final_title = complaint.title
    if ai_department and f"{final_subcategory} at {final_department}" != complaint.title:
        final_title = f"{final_subcategory} at {final_department}"

    print(f"[COMPLAINT] Final → dept={final_department}, sub={final_subcategory}, severity={ai_severity}, confidence={ai_confidence}%")

    try:
        new_complaint = models.Complaint(
            title=final_title,
            description=complaint.description,
            image_url=complaint.image_url,
            voice_url=complaint.voice_url,
            latitude=complaint.latitude,
            longitude=complaint.longitude,
            reporter_id=current_user.id,
            department=final_department,
            issue_type=final_subcategory,
            severity_score=ai_severity,
            confidence_score=ai_confidence,
            department_suggested=ai_department or final_department,
        )
        db.add(new_complaint)
        db.commit()
        db.refresh(new_complaint)
        print(f"[COMPLAINT] ✅ Saved. ID={new_complaint.id} | dept={new_complaint.department} | issue={new_complaint.issue_type} | severity={new_complaint.severity_score}")
        return new_complaint
    except Exception as e:
        print(f"[COMPLAINT] ❌ DB Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/complaints/", response_model=list[schemas.ComplaintAIResponse])
def get_all_complaints(db: Session = Depends(get_db)):
    complaints = db.query(models.Complaint).all()
    print(f"[FETCH] All complaints count: {len(complaints)}")
    for c in complaints:
        print(f"  - ID:{c.id} | image_url:{c.image_url} | voice_url:{c.voice_url}")
    return complaints

@app.get("/complaints/me", response_model=list[schemas.ComplaintAIResponse])
def get_my_complaints(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    complaints = db.query(models.Complaint).filter(models.Complaint.reporter_id == current_user.id).all()
    print(f"[FETCH] Complaints for user {current_user.phone_number}: {len(complaints)}")
    for c in complaints:
        print(f"  - ID:{c.id} | image_url:{c.image_url} | voice_url:{c.voice_url}")
    return complaints


@app.get("/workers/", response_model=list[schemas.Worker])
def get_all_workers(db: Session = Depends(get_db)):
    # If table empty, add mock ones
    workers = db.query(models.Worker).all()
    if not workers:
        mock_workers = [
            models.Worker(name="Rajinder Kumar", department="Roads & Bridges", status="Active", phone="+91 9876543100", location="Sector 14", rating=4.8),
            models.Worker(name="Suresh Patil", department="Waste Mgmt", status="On Leave", phone="+91 9876543101", location="N/A", rating=4.9),
            models.Worker(name="Amit Sharma", department="Water Supply", status="Assigned", phone="+91 9876543102", location="MG Road", rating=4.5),
        ]
        db.add_all(mock_workers)
        db.commit()
        workers = db.query(models.Worker).all()
    return workers
@app.patch("/complaints/{complaint_id}/status")
def update_complaint_status(complaint_id: int, status_update: dict, db: Session = Depends(get_db)):
    comp = db.query(models.Complaint).filter(models.Complaint.id == complaint_id).first()
    if not comp:
        raise HTTPException(status_code=404, detail="Complaint not found")
    
    new_status = status_update.get("status")
    if new_status:
        comp.status = new_status
        db.commit()
        db.refresh(comp)
    return comp
