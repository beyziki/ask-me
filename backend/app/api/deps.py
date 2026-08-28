"""Ortak API bağımlılıkları."""
from fastapi import Header, HTTPException

from backend.app.db.base import get_db  # re-export

__all__ = ["get_db", "get_current_user_id"]


def get_current_user_id(x_user_id: int = Header(..., description="Giriş yapan kullanıcının id'si")) -> int:
    """MVP aşamasında basit header-bazlı kimlik.

    TODO (Hafta 3): JWT tabanlı gerçek oturum yönetimine geçilecek.
    """
    if x_user_id <= 0:
        raise HTTPException(status_code=401, detail="Geçersiz kullanıcı")
    return x_user_id
