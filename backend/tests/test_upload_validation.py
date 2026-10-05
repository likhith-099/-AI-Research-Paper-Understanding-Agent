from io import BytesIO

import pytest
from fastapi import HTTPException, UploadFile

from api.main import analyze_upload


@pytest.mark.asyncio
async def test_analyze_upload_rejects_empty_file():
    upload = UploadFile(filename="empty.pdf", file=BytesIO(b""))

    with pytest.raises(HTTPException) as error:
        await analyze_upload(upload)

    assert error.value.status_code == 400
    assert error.value.detail == "Uploaded PDF is empty."