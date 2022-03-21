terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "4.13.0"
    }
  }
}

provider "google" {
  credentials = file("${path.module}/../credentials/youyaku-ai-service-account.json")
  project     = lookup(var.env_parameters, "GOOGLE_PROJECT_ID")
  region      = lookup(var.env_parameters, "GOOGLE_PREDICTION_LOCATION")
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
        env {
          name  = "PORT"
          value = lookup(var.env_parameters, "DASHBORAD_PORT")
        }
        env {
          name  = "API_HOST"
          value = lookup(var.env_parameters, "API_HOST")
        }
        env {
          name  = "API_PORT"
          value = lookup(var.env_parameters, "API_PORT")
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service" "api_gateway" {
  name     = "cloudrun-srv"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "us.gcr.io/youyaku-ai/api_gateway"
        env {
          name  = "PORT"
          value = lookup(var.env_parameters, "API_PORT")
        }
        env {
          name  = "QUEUE_HOST"
          value = lookup(var.env_parameters, "QUEUE_HOST")
        }
        env {
          name  = "QUEUE_NAME"
          value = lookup(var.env_parameters, "QUEUE_NAME")
        }
        env {
          name  = "QUEUE_PORT"
          value = lookup(var.env_parameters, "QUEUE_PORT")
        }
        env {
          name  = "DB_HOST"
          value = lookup(var.env_parameters, "DB_HOST")
        }
        env {
          name  = "DB_PORT"
          value = lookup(var.env_parameters, "DB_PORT")
        }
        env {
          name  = "DB_NAME"
          value = lookup(var.env_parameters, "DB_NAME")
        }
        env {
          name  = "DB_TYPE"
          value = lookup(var.env_parameters, "DB_TYPE")
        }
        env {
          name  = "DB_USERNAME"
          value = lookup(var.env_parameters, "DB_USERNAME")
        }
        env {
          name  = "DB_PASSWORD"
          value = lookup(var.env_parameters, "DB_PASSWORD")
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_cloud_run_service" "summarizer_processor" {
  name     = "cloudrun-srv"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "us.gcr.io/youyaku-ai/summarizer_processor"
        env {
          name  = "QUEUE_HOST"
          value = lookup(var.env_parameters, "QUEUE_HOST")
        }
        env {
          name  = "QUEUE_NAME"
          value = lookup(var.env_parameters, "QUEUE_NAME")
        }
        env {
          name  = "QUEUE_PORT"
          value = lookup(var.env_parameters, "QUEUE_PORT")
        }
        env {
          name  = "DB_HOST"
          value = lookup(var.env_parameters, "DB_HOST")
        }
        env {
          name  = "DB_PORT"
          value = lookup(var.env_parameters, "DB_PORT")
        }
        env {
          name  = "DB_NAME"
          value = lookup(var.env_parameters, "DB_NAME")
        }
        env {
          name  = "DB_TYPE"
          value = lookup(var.env_parameters, "DB_TYPE")
        }
        env {
          name  = "DB_USERNAME"
          value = lookup(var.env_parameters, "DB_USERNAME")
        }
        env {
          name  = "DB_PASSWORD"
          value = lookup(var.env_parameters, "DB_PASSWORD")
        }
        env {
          name  = "SUMMARIZER_INTERNAL_API_LOCAL_HOST"
          value = lookup(var.env_parameters, "SUMMARIZER_INTERNAL_API_LOCAL_HOST")
        }
        env {
          name  = "SUMMARIZER_INTERNAL_API_LOCAL_PORT"
          value = lookup(var.env_parameters, "SUMMARIZER_INTERNAL_API_LOCAL_PORT")
        }
        env {
          name  = "GOOGLE_PROJECT_ID"
          value = lookup(var.env_parameters, "GOOGLE_PROJECT_ID")
        }
        env {
          name  = "GOOGLE_PREDICTION_LOCATION"
          value = lookup(var.env_parameters, "GOOGLE_PREDICTION_LOCATION")
        }
        env {
          name  = "GOOGLE_PREDICTION_ENDPOINT"
          value = lookup(var.env_parameters, "GOOGLE_PREDICTION_ENDPOINT")
        }
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

resource "google_pubsub_topic" "summarizer_queue" {
    name = lookup(var.env_parameters, "QUEUE_NAME")

    message_retention_duration = "86600s"
}

resource "google_pubsub_subscription" "summarizer_queue_subscription" {
    name = "${lookup(var.env_parameters, "QUEUE_NAME")}-sub"
    topic = google_pubsub_topic.summarizer_queue.name
    ack_deadline_seconds = 20
}