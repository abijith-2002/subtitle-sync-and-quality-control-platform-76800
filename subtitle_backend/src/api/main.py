from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from .models import (
    UserAuthRequest, AuthTokenResponse, UploadMediaResponse,
    SyncStatus, ValidationReport, ValidationReportItem, DashboardSummary
)
from .processing import (
    auto_align_subtitles, check_reading_speed, detect_overlap, check_frame_rate, check_language, spellcheck_subtitles
)
from .auth import (
    users_db, get_password_hash, authenticate_user, create_access_token, get_current_user
)
import os
import uuid
import shutil

app = FastAPI(
    title="Subtitle Sync & Quality Control API",
    description="API for uploading media/subtitles, performing sync and validation, and retrieving quality control reports.",
    version="1.0.0",
    openapi_tags=[
        {"name": "auth", "description": "User authentication endpoints"},
        {"name": "media", "description": "Media/video upload & retrieval"},
        {"name": "subtitle", "description": "Subtitle file upload, processing"},
        {"name": "sync", "description": "Subtitle sync and validation jobs"},
        {"name": "report", "description": "Analysis, validation & reporting"},
        {"name": "dashboard", "description": "Dashboard APIs"},
    ]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ############ Dummy in-memory data store for jobs/results ##################
UPLOAD_ROOT = "/tmp/subtitle_files"
os.makedirs(UPLOAD_ROOT, exist_ok=True)

media_db = {}     # media_id: {filename, type, user}
subtitle_db = {}  # sub_id: {filename, language, user, media_id}
sync_jobs = {}    # sync_id: {media_id, sub_id, user, status, report}

######################################
#              AUTH                  #
######################################

# PUBLIC_INTERFACE
@app.post("/api/auth/signup", tags=["auth"], response_model=AuthTokenResponse)
def signup(req: UserAuthRequest):
    """Signup endpoint for new user registration."""
    if req.email in users_db:
        raise HTTPException(status_code=409, detail="User already exists")
    users_db[req.email] = {"email": req.email, "password": get_password_hash(req.password)}
    access_token = create_access_token({"sub": req.email})
    return AuthTokenResponse(access_token=access_token, token_type="bearer")

# PUBLIC_INTERFACE
@app.post("/api/auth/login", tags=["auth"], response_model=AuthTokenResponse)
def login(req: UserAuthRequest):
    """Login endpoint for existing users."""
    user = authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token({"sub": req.email})
    return AuthTokenResponse(access_token=access_token, token_type="bearer")

# PUBLIC_INTERFACE
@app.post("/api/auth/token", tags=["auth"], response_model=AuthTokenResponse)
def token(form_data: OAuth2PasswordRequestForm = Depends()):
    """OAuth2 password token endpoint."""
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    token = create_access_token({"sub": form_data.username})
    return AuthTokenResponse(access_token=token, token_type="bearer")

######################################
#             UPLOAD                 #
######################################

# PUBLIC_INTERFACE
@app.post("/api/upload/media", tags=["media"], response_model=UploadMediaResponse)
async def upload_media(
    file: UploadFile = File(...),
    current_user=Depends(get_current_user)
):
    """Upload an audio or video file for subtitle processing."""
    media_id = str(uuid.uuid4())
    filename = f"{media_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_ROOT, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    media_db[media_id] = {"filename": filename, "type": file.content_type, "user": current_user["email"]}
    return UploadMediaResponse(media_id=media_id, filename=filename, detail="uploaded")

# PUBLIC_INTERFACE
@app.post("/api/upload/subtitle", tags=["subtitle"], response_model=UploadMediaResponse)
async def upload_subtitle(
    file: UploadFile = File(...),
    media_id: str = Form(...),
    language: str = Form(...),
    current_user=Depends(get_current_user),
):
    """Upload subtitle for a media file."""
    if media_id not in media_db:
        raise HTTPException(status_code=404, detail="Related media not found")
    sub_id = str(uuid.uuid4())
    filename = f"{sub_id}_{file.filename}"
    file_path = os.path.join(UPLOAD_ROOT, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    subtitle_db[sub_id] = {
        "filename": filename, "language": language, "user": current_user["email"], "media_id": media_id
    }
    return UploadMediaResponse(media_id=sub_id, filename=filename, detail="uploaded")

#######################################
#         SYNC & VALIDATE             #
#######################################

# PUBLIC_INTERFACE
@app.post("/api/sync/align", tags=["sync"], response_model=SyncStatus)
def sync_subtitle_with_media(
    media_id: str = Form(...),
    sub_id: str = Form(...),
    current_user=Depends(get_current_user)
):
    """Start auto-alignment and validation sync for subtitle + media."""
    if media_id not in media_db or sub_id not in subtitle_db:
        raise HTTPException(status_code=404, detail="Media or subtitle not found.")
    sync_id = str(uuid.uuid4())
    media_path = os.path.join(UPLOAD_ROOT, media_db[media_id]['filename'])
    sub_path = os.path.join(UPLOAD_ROOT, subtitle_db[sub_id]['filename'])
    # Read subtitle content
    with open(sub_path, "r") as f:
        sub_content = f.read()
    # Dummy auto-alignment
    aligned = auto_align_subtitles(sub_content, media_path)
    # Save aligned copy, update subtitle_db
    aligned_filename = f"{sub_id}_aligned.srt"
    aligned_path = os.path.join(UPLOAD_ROOT, aligned_filename)
    with open(aligned_path, "w") as f:
        f.write(aligned)
    subtitle_db[sub_id]['aligned_filename'] = aligned_filename

    # Validation/Reporting
    reading_speed = check_reading_speed(aligned)
    overlap = detect_overlap(aligned)
    frame_rate = check_frame_rate(media_path)
    language = check_language(aligned, subtitle_db[sub_id]['language'])
    spelling = spellcheck_subtitles(aligned)
    report = [
        ValidationReportItem(type='reading_speed', passed=reading_speed['passed'], highlight_segments=reading_speed.get('highlight_segments')),
        ValidationReportItem(type='overlap', passed=overlap['passed'], highlight_segments=overlap.get('highlight_segments')),
        ValidationReportItem(type='frame_rate', passed=frame_rate['passed']),
        ValidationReportItem(type='language', passed=language['passed'], detail=language.get('detail')),
        ValidationReportItem(type='spelling', passed=spelling['passed'], highlight_segments=spelling.get('highlight_segments')),
    ]

    sync_jobs[sync_id] = {
        "media_id": media_id,
        "sub_id": sub_id,
        "user": current_user["email"],
        "status": "COMPLETED",
        "report": report
    }
    return SyncStatus(sync_id=sync_id, status="COMPLETED", detail="Alignment and validation complete")

#######################################
#         VALIDATION/REPORT            #
#######################################

# PUBLIC_INTERFACE
@app.get("/api/report/{sync_id}", tags=["report"], response_model=ValidationReport)
def get_validation_report(sync_id: str, current_user=Depends(get_current_user)):
    """Get the validation report for a subtitle/media sync job."""
    job = sync_jobs.get(sync_id)
    if not job or job["user"] != current_user["email"]:
        raise HTTPException(status_code=404, detail="Sync job/report not found for user")
    return ValidationReport(sync_id=sync_id, results=job["report"])

# PUBLIC_INTERFACE
@app.get("/api/subtitle/aligned/{sub_id}", tags=["subtitle"])
def get_aligned_subtitle_file(sub_id: str, current_user=Depends(get_current_user)):
    """Download aligned subtitle file."""
    sub_info = subtitle_db.get(sub_id)
    if not sub_info or sub_info["user"] != current_user["email"]:
        raise HTTPException(status_code=404, detail="Subtitle not found")
    fname = sub_info.get("aligned_filename")
    aligned_path = os.path.join(UPLOAD_ROOT, fname) if fname else None
    if not fname or not os.path.isfile(aligned_path):
        raise HTTPException(status_code=404, detail="Aligned subtitle not found")
    return FileResponse(aligned_path, filename=fname)

#######################################
#         DASHBOARD                   #
#######################################

# PUBLIC_INTERFACE
@app.get("/api/dashboard/jobs", tags=["dashboard"], response_model=DashboardSummary)
def list_user_jobs(current_user=Depends(get_current_user)):
    """Get all jobs (media/sub/validation) for the current user."""
    jobs = []
    for sync_id, job in sync_jobs.items():
        if job["user"] == current_user["email"]:
            jobs.append(SyncStatus(sync_id=sync_id, status=job["status"], detail=""))
    return DashboardSummary(jobs=jobs)

@app.get("/", tags=["health"])
def health_check():
    """Health check endpoint."""
    return {"message": "Healthy"}
