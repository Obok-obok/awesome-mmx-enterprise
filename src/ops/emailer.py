from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import List, Optional


def _get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def send_email_with_attachment(
    subject: str,
    body_text: str,
    attachment_path: Path,
    *,
    smtp_host: Optional[str] = None,
    smtp_port: Optional[int] = None,
    smtp_user: Optional[str] = None,
    smtp_pass: Optional[str] = None,
    mail_from: Optional[str] = None,
    mail_to: Optional[List[str]] = None,
    use_tls: Optional[bool] = None,
) -> None:
    """Send a simple email with a single attachment via SMTP.

    Config via env vars (recommended):
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM, SMTP_TO (comma-separated), SMTP_TLS(true/false)
    """

    smtp_host = smtp_host or _get_env("SMTP_HOST")
    smtp_port = smtp_port or int(_get_env("SMTP_PORT", "587"))
    smtp_user = smtp_user or _get_env("SMTP_USER")
    smtp_pass = smtp_pass or _get_env("SMTP_PASS")
    mail_from = mail_from or _get_env("SMTP_FROM", smtp_user)
    to_raw = ",".join(mail_to) if mail_to else _get_env("SMTP_TO")
    mail_to = [x.strip() for x in (to_raw or "").split(",") if x.strip()]
    use_tls = use_tls if use_tls is not None else (_get_env("SMTP_TLS", "true").lower() in ("1", "true", "yes"))

    if not smtp_host or not mail_from or not mail_to:
        raise ValueError("Missing SMTP settings. Set SMTP_HOST/SMTP_FROM/SMTP_TO (and usually SMTP_USER/SMTP_PASS).")

    if not attachment_path.exists():
        raise FileNotFoundError(str(attachment_path))

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = ", ".join(mail_to)
    msg.set_content(body_text)

    data = attachment_path.read_bytes()
    msg.add_attachment(data, maintype="application", subtype="pdf", filename=attachment_path.name)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as s:
        if use_tls:
            s.starttls()
        if smtp_user and smtp_pass:
            s.login(smtp_user, smtp_pass)
        s.send_message(msg)
