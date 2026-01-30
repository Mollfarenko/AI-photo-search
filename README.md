# AI Photo Search

A **private, AI-powered photo search system** for personal photo collections. The project enables natural-language and image-based search using CLIP embeddings, metadata filtering, and an LLM reasoning layer — all running securely inside AWS.

The system is designed to be **non-public by default**, accessible only from trusted devices via secure tunneling (SSM) or a private network (Tailscale).

---

## Key Features

* 🔍 Semantic photo search using **CLIP embeddings** (image ↔ text)
* 🧠 LLM-assisted query understanding (OpenAI GPT-4 mini)
* 🖼️ Search by **text or image** with optional metadata filters
* ☁️ Scalable ingestion via **S3 → Lambda → SQS**
* 📦 Persistent vector storage using **Chroma + EBS**
* 🔐 Private access via **AWS SSM** or **Tailscale**
* 🐳 Fully containerized with Docker

---

## High-Level Architecture

```
Phone / Browser
      │
      │  (SSM tunnel or Tailscale)
      ▼
FastAPI (Docker, EC2)
      │
      ├── LLM Agent (query reasoning)
      ├── CLIP Text Embeddings
      ├── CLIP Image Embeddings
      ▼
Chroma Vector DB (EBS)
```

---

## Photo Ingestion Pipeline

### 1. Upload

Photos are uploaded (currently **manually**) to an **S3 uploads folder**.

> In the future, this can be automated (e.g. mobile photo gallery sync).

---

### 2. Lambda Processing

An S3 event triggers a Lambda function that:

* Extracts EXIF and basic metadata
* Generates thumbnails
* Stores processed images back to S3
* Sends metadata + S3 keys to **SQS**

---

### 3. Embedding Worker (EC2)

A long-running worker on EC2 consumes SQS messages:

* Downloads images from S3
* Generates **CLIP image embeddings**
* Stores embeddings + metadata in **Chroma**
* Persists everything on **EBS** (not ephemeral)

This decoupling allows ingestion to scale independently from search.

---

## Search & API Layer

A **FastAPI** application runs inside Docker on EC2 and serves both:

* Backend API
* Frontend UI

### Main Endpoints

* `/` – Frontend UI
* `/search/text` – Semantic text search
* `/search/image` – Image similarity search
* `/health` – Health check

The frontend and backend are intentionally served from the **same FastAPI server** to avoid CORS, localhost, and multi-server complexity.

---

## LLM Reasoning Layer

The system uses **OpenAI GPT-4 mini** as a *reasoning and orchestration layer*, not as a source of truth.

The LLM:

* Translates and normalizes user queries
* Decides when to:

  * perform vector similarity search
  * apply metadata filters
  * or combine both
* Calls internal tools with **strict safety rules**:

### LLM Constraints

* ❌ Never hallucinate photos
* ❌ Never invent metadata
* ❌ Never describe images not returned by tools
* ✅ Only summarize and explain tool results

This keeps the system deterministic and trustworthy.

---

## Security & Access Model

This project is **private by design**.

* ❌ No public HTTP endpoints
* ❌ No exposed authentication UI
* ❌ No public IP required

### Access Options

#### Option 1: AWS SSM (Desktop)

* Secure port forwarding using AWS credentials
* No SSH keys exposed
* Ideal for development and debugging

#### Option 2 (Recommended): Tailscale

* Creates a private WireGuard-based network
* Works across:

  * laptop
  * mobile phone
  * EC2
* Access the app via:

```
http://<tailscale-ip>:8000
```

This is the **cleanest solution** for private, multi-device access.

---

## Project Structure

```
/home/ec2-user/app
├── agents/                 # LLM agent orchestration
│   └── agent_runtime.py
├── embeddings/             # CLIP models & embedding logic
│   ├── clip_model.py
│   ├── image_embedder.py
│   └── text_embedder.py
├── worker/                 # SQS-driven embedding worker
│   └── embed_images_worker.py
├── storage/                # Vector store abstraction (Chroma)
│   └── chroma_store.py
├── tools/                  # Search & filtering tools
│   ├── text_search.py
│   ├── image_search.py
│   ├── metadata_filter.py
│   └── unified_search.py
├── llm/                    # LLM client & system prompt
│   └── llm.py
├── utilities/              # Helpers (S3 URLs, viewers)
│   └── url_generator.py
├── inspectors/             # Debug & inspection utilities
│   └── chroma_inspector.py
├── data/chroma/            # Persistent vector DB (EBS-backed)
├── tmp/images/             # Temporary image storage
├── entrypoint/             # CLI utilities
│   └── cli_agent.py
├── frontend/               # Static frontend (served by FastAPI)
└── requirements.txt
```

---

## Development Notes

* Docker is used for repeatable deployments
* EBS ensures embeddings persist across restarts
* The system currently supports **manual ingestion** only

---

## Roadmap

* 📱 Automatic phone gallery sync
* 🔄 Background re-embedding on metadata updates
* 🧑 Face clustering & people search
* 🕰️ Timeline-based photo exploration
* 📐 Improved mobile-first UI

---

## Philosophy

This project prioritizes:

* **Privacy over exposure**
* **Reasoned AI over hallucinations**
* **Simple infrastructure over over-engineering**

It is intentionally built to scale **only when needed**.
