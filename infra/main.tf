terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "3.5.0"
    }
  }
}

provider "google" {
  credentials = file("${path.module}/../credentials/youyaku-ai-service-account.json")
  project     = "youyaku-ai"
  region      = "us-central1"
  zone        = "us-central1-c"
}

variable "env_parameters" {
  type = map(string)
}

resource "google_cloud_run_service" "dashboard" {
  name     = "cloudrun-srv"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "us.gcr.io/youyaku-ai/dashboard"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}
