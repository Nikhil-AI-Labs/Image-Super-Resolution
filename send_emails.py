import smtplib
import getpass
import os
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# ── Configuration ──────────────────────────────────────────────────
EMAILS_FILE = r'd:\super-resolution\ntire_emails.txt'
FACTSHEET_PDF = r'd:\super-resolution\NTIRE.pdf'
FACTSHEET_SOURCE_DIR = r'd:\super-resolution\factsheet'   # directory with .tex / .bib sources

# Set to True to send ONLY to yourself for a test run first
TEST_MODE = True
TEST_RECIPIENT = "nikhilpathaksvnit@gmail.com"


def parse_email_file(filename):
    """Parse the email text file to extract subject, recipients, CC, and body."""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()

    subject = ""
    to_list = []
    cc_list = []
    body_lines = []
    in_body = False

    for line in content.split('\n'):
        if in_body:
            body_lines.append(line)
        elif line.startswith("Subject:"):
            subject = line.replace("Subject:", "").strip()
        elif line.startswith("To:"):
            raw = line.replace("To:", "").strip()
            to_list = [e.strip() for e in raw.split(",") if e.strip()]
        elif line.startswith("CC:"):
            raw = line.replace("CC:", "").strip()
            cc_list = [e.strip() for e in raw.split(",") if e.strip()]
        elif line.startswith("Body:"):
            in_body = True

    body = "\n".join(body_lines).strip()
    return subject, to_list, cc_list, body


def collect_attachments():
    """Collect the factsheet PDF and any .tex/.bib source files."""
    attachments = []

    # 1) Compiled factsheet PDF
    if os.path.exists(FACTSHEET_PDF):
        attachments.append(FACTSHEET_PDF)
    else:
        print(f"  [WARNING] Factsheet PDF not found: {FACTSHEET_PDF}")

    # 2) .tex and .bib source files from factsheet directory
    if os.path.isdir(FACTSHEET_SOURCE_DIR):
        for fname in os.listdir(FACTSHEET_SOURCE_DIR):
            if fname.endswith(('.tex', '.bib')):
                attachments.append(os.path.join(FACTSHEET_SOURCE_DIR, fname))
    else:
        print(f"  [WARNING] Factsheet source dir not found: {FACTSHEET_SOURCE_DIR}")

    return attachments


def send_email():
    print("─── NTIRE 2026 Factsheet Submission ───")
    print(f"Reading email from: {EMAILS_FILE}\n")

    if not os.path.exists(EMAILS_FILE): #marc ywii ffwa pkrk
        print(f"Error: Email file not found: {EMAILS_FILE}")
        return

    subject, to_list, cc_list, body = parse_email_file(EMAILS_FILE)

    if not subject or not to_list or not body:
        print("Error: Could not parse subject / recipients / body from email file.")
        return

    # ── Collect attachments ──
    attachments = collect_attachments()

    print(f"Subject : {subject}")
    print(f"To      : {', '.join(to_list)}")
    print(f"CC      : {', '.join(cc_list)}")
    print(f"Attachments ({len(attachments)}):")
    for a in attachments:
        print(f"  • {os.path.basename(a)}")

    # ── Test-mode override ──
    if TEST_MODE:
        print(f"\n[TEST MODE] Redirecting everything to: {TEST_RECIPIENT}")
        to_list = [TEST_RECIPIENT]
        cc_list = []

    # ── Credentials ──
    print("\nTo send, log in with your Gmail (use App Password if 2FA is on).")
    user_email = input("Gmail address : ")
    password = getpass.getpass("App Password  : ")

    # ── Build MIME message ──
    msg = MIMEMultipart()
    msg['From'] = user_email
    msg['To'] = ", ".join(to_list)
    if cc_list:
        msg['Cc'] = ", ".join(cc_list)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    # Attach files
    for filepath in attachments:
        with open(filepath, "rb") as f:
            part = MIMEApplication(f.read(), _subtype="pdf" if filepath.endswith('.pdf') else "octet-stream")
            part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(filepath))
            msg.attach(part)

    # ── Send ──
    all_recipients = to_list + cc_list
    try:
        print("\nConnecting to Gmail SMTP …")
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(user_email, password)
        print("Logged in ✓")

        server.sendmail(user_email, all_recipients, msg.as_string())
        print("Email sent successfully ✓")

        server.quit()
    except smtplib.SMTPAuthenticationError:
        print("\nError: Authentication failed. Check your email / App Password.")
    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    send_email()
