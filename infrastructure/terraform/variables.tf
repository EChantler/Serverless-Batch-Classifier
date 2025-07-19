variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "aurora_master_username" {
  description = "Aurora master username"
  type        = string
}

variable "aurora_master_password" {
  description = "Aurora master password"
  type        = string
  sensitive   = true
}

variable "aurora_min_capacity" {
  description = "Aurora min capacity"
  type        = number
  default     = 2
}

variable "aurora_max_capacity" {
  description = "Aurora max capacity"
  type        = number
  default     = 8
}

variable "aurora_auto_pause_seconds" {
  description = "Aurora seconds until auto pause"
  type        = number
  default     = 300
}

variable "s3_bucket_name" {
  description = "S3 bucket name for Lambda access"
  type        = string
}

variable "lambda_image_uri" {
  description = "ECR image URI for Lambda"
  type        = string
}
