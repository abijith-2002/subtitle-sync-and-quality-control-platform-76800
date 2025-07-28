from typing import Optional, List, Dict
from pydantic import BaseModel, Field

# PUBLIC_INTERFACE
class UserAuthRequest(BaseModel):
    """Request model for user login/signup."""
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")

# PUBLIC_INTERFACE
class AuthTokenResponse(BaseModel):
    """Response model for authentication token."""
    access_token: str = Field(..., description="JWT Access Token")
    token_type: str = Field(..., description="Token type, typically 'bearer'")

# PUBLIC_INTERFACE
class UploadSubtitleRequest(BaseModel):
    """Request model meta-info for uploading subtitles."""
    media_id: str = Field(..., description="ID of related audio/video media")
    language: str = Field(..., description="Language code of the subtitle, e.g., 'en', 'es'")

# PUBLIC_INTERFACE
class UploadMediaResponse(BaseModel):
    """Response for media/subtitle upload."""
    media_id: str = Field(..., description="ID of uploaded media or subtitle")
    filename: str = Field(..., description="Filename on server")
    detail: str = Field(..., description="Status message, e.g., 'uploaded'")

# PUBLIC_INTERFACE
class SyncStatus(BaseModel):
    """Subtitle/media synchronization job status."""
    sync_id: str = Field(..., description="Sync job identifier")
    status: str = Field(..., description="Status message, e.g., PENDING, COMPLETED, FAILED")
    detail: Optional[str] = Field(None, description="Optional information on the process")

# PUBLIC_INTERFACE
class ValidationReportItem(BaseModel):
    """Item in the subtitle validation report."""
    type: str = Field(..., description="Type of check (e.g. 'speed', 'overlap', 'frame_rate', 'spelling')")
    passed: bool = Field(..., description="Whether this check passed")
    detail: Optional[str] = Field(None, description="Extra detail or error messages for this check")
    highlight_segments: Optional[List[Dict]] = Field(None, description="Segments/lines to highlight for problems")

# PUBLIC_INTERFACE
class ValidationReport(BaseModel):
    """Validation report for a subtitle/media sync job."""
    sync_id: str = Field(..., description="Sync job identifier")
    results: List[ValidationReportItem] = Field(..., description="List of check results")

# PUBLIC_INTERFACE
class DashboardSummary(BaseModel):
    """Basic dashboard summary of user's jobs."""
    jobs: List[SyncStatus] = Field(..., description="List of all jobs and status owned by the user")
