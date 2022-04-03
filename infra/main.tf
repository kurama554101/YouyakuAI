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

resource "google_compute_network" "vpc" {
  name                    = "cloudrun-network"
  provider                = google
  auto_create_subnetworks = true
}

resource "google_compute_global_address" "private_db_address" {
  name          = "private-ip-address"
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 16
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "private_vpc_connection" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_db_address.name]
}

resource "google_sql_user" "summarizer_db_user" {
  name     = lookup(var.env_parameters, "DB_USERNAME")
  instance = google_sql_database_instance.summarizer_db_mysql.name
  password = lookup(var.env_parameters, "DB_PASSWORD")
}

resource "google_sql_database_instance" "summarizer_db_mysql" {
  name = lookup(var.env_parameters, "DB_HOST")
  depends_on = [google_service_networking_connection.private_vpc_connection]
  database_version = "MYSQL_8_0"
  region = "us-central1"
  settings {
    tier = "db-f1-micro"
    ip_configuration {
      ipv4_enabled    = false
      private_network = google_compute_network.vpc.id
    }
  }
}

resource "google_service_account" "dashboard_account" {
  account_id = "dashboard-account"
  display_name = "dashboard service account"
}

resource "google_cloud_run_service" "dashboard" {
  name     = "cloudrun-srv"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "us.gcr.io/youyaku-ai/dashboard"
        ports {
            container_port = lookup(var.env_parameters, "DASHBORAD_PORT")
        }
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
      service_account_name = google_service_account.dashboard_account.email
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

data "google_iam_policy" "dashboard_policy" {
  binding {
    role = "roles/run.invoker"
    members = [
      "allUsers",
    ]
  }
}

resource "google_cloud_run_service_iam_policy" "dashboard_policy" {
  location    = google_cloud_run_service.dashboard.location
  project     = google_cloud_run_service.dashboard.project
  service     = google_cloud_run_service.dashboard.name

  policy_data = data.google_iam_policy.dashboard_policy.policy_data
}

resource "google_vpc_access_connector" "vpc_connector" {
  name          = "vpc-connector"
  ip_cidr_range = "10.14.0.0/28"
  network       = google_compute_network.vpc.name
}

resource "google_service_account" "api_gateway_account" {
  account_id = "api-gateway-account"
  display_name = "api gateway service account"
}

resource "google_cloud_run_service" "api_gateway" {
  name     = "cloudrun-srv"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "us.gcr.io/youyaku-ai/api_gateway"
        ports {
            container_port = lookup(var.env_parameters, "API_PORT")
        }
        env {
          name  = "PORT"
          value = lookup(var.env_parameters, "API_PORT")
        }
        env {
          name  = "QUEUE_NAME"
          value = lookup(var.env_parameters, "QUEUE_NAME")
        }
        env {
          name  = "QUEUE_TYPE"
          value = lookup(var.env_parameters, "QUEUE_TYPE")
        }
        env {
          name  = "DB_HOST"
          value = google_compute_global_address.private_db_address.address
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
          value = google_sql_user.summarizer_db_user.name
        }
        env {
          name  = "DB_PASSWORD"
          value = google_sql_user.summarizer_db_user.password
        }
      }
      service_account_name = google_service_account.api_gateway_account.email
    }
    metadata {
      annotations = {
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.vpc_connector.name
        "run.googleapis.com/vpc-access-egress"    = "private-ranges-only"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

data "google_iam_policy" "api_gateway_policy" {
  binding {
    role = "roles/run.invoker"
    members = [
      google_service_account.dashboard_account.email
    ]
  }
  binding {
    role = "roles/pubsub.editor"
    members = [
      google_service_account.api_gateway_account.email
    ]
  }
}

resource "google_cloud_run_service_iam_policy" "api_gateway_policy" {
  location    = google_cloud_run_service.api_gateway.location
  project     = google_cloud_run_service.api_gateway.project
  service     = google_cloud_run_service.api_gateway.name

  policy_data = data.google_iam_policy.api_gateway_policy.policy_data
}

resource "google_service_account" "summarizer_processor_account" {
  account_id = "summarizer-processor-account"
  display_name = "summarizer processor service account"
}

resource "google_cloud_run_service" "summarizer_processor" {
  name     = "cloudrun-srv"
  location = "us-central1"

  template {
    spec {
      containers {
        image = "us.gcr.io/youyaku-ai/summarizer_processor"
        env {
          name  = "QUEUE_NAME"
          value = lookup(var.env_parameters, "QUEUE_NAME")
        }
        env {
          name  = "QUEUE_TYPE"
          value = lookup(var.env_parameters, "QUEUE_TYPE")
        }
        env {
          name  = "DB_HOST"
          value = google_compute_global_address.private_db_address.address
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
          value = google_sql_user.summarizer_db_user.name
        }
        env {
          name  = "DB_PASSWORD"
          value = google_sql_user.summarizer_db_user.password
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
      service_account_name = google_service_account.summarizer_processor_account.email
    }
    metadata {
      annotations = {
        "run.googleapis.com/vpc-access-connector" = google_vpc_access_connector.vpc_connector.name
        "run.googleapis.com/vpc-access-egress"    = "private-ranges-only"
      }
    }
  }

  traffic {
    percent         = 100
    latest_revision = true
  }
}

data "google_iam_policy" "summarizer_processor_policy" {
  binding {
    role = "roles/pubsub.subscriber"
    members = [
      google_service_account.summarizer_processor_account.email
    ]
  }
}

resource "google_cloud_run_service_iam_policy" "summarizer_processor_policy" {
  location    = google_cloud_run_service.summarizer_processor.location
  project     = google_cloud_run_service.summarizer_processor.project
  service     = google_cloud_run_service.summarizer_processor.name

  policy_data = data.google_iam_policy.summarizer_processor_policy.policy_data
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
