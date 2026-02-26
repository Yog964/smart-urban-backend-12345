from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from database import engine, Base, SessionLocal
import models, schemas, utils

def get_db():
    print("Opening DB session...")
    db = SessionLocal()
    try:
        yield db
    finally:
        print("Closing DB session.")
        db.close()
from fastapi.middleware.cors import CORSMiddleware
import os

Base.metadata.create_all(bind=engine)

app = FastAPI(title="UrbanSathi Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, specify your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

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


# Fake directory for uploads for now. Vercel is read-only except /tmp
UPLOAD_DIR = "/tmp/uploads" if os.environ.get("VERCEL") else "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/", response_model=dict)
async def upload_file(file: UploadFile = File(...)):
    import uuid
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_location = f"{UPLOAD_DIR}/{unique_filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    # Return as path so frontend logic doesn't break
    return {"image_url": f"uploads/{unique_filename}"}


@app.post("/complaints/", response_model=schemas.ComplaintAIResponse)
def create_complaint(complaint: schemas.ComplaintCreate, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # MOCK AI Service Call
    ai_issue_type = "Pothole"
    ai_severity = 7.5
    ai_confidence = 88.0
    ai_dept = "Roads & Bridges"
    
    new_complaint = models.Complaint(
        title=complaint.title,
        description=complaint.description,
        image_url=complaint.image_url,
        voice_url=complaint.voice_url,
        latitude=complaint.latitude,
        longitude=complaint.longitude,
        reporter_id=current_user.id,
        department=complaint.department,
        issue_type=complaint.subcategory, # Maps to subcategory select
        severity_score=ai_severity,
        confidence_score=ai_confidence,
        department_suggested=ai_dept # Predicted
    )
    
    db.add(new_complaint)
    db.commit()
    db.refresh(new_complaint)
    
    return new_complaint

@app.get("/complaints/", response_model=list[schemas.ComplaintAIResponse])
def get_all_complaints(db: Session = Depends(get_db)):
    return db.query(models.Complaint).all()

@app.get("/complaints/me", response_model=list[schemas.ComplaintAIResponse])
def get_my_complaints(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    return db.query(models.Complaint).filter(models.Complaint.reporter_id == current_user.id).all()

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
