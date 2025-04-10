variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name (e.g. dev, staging, prod)"
  type        = string
  default     = "prod"
}

variable "cluster_name" {
  description = "Name of the EKS cluster"
  type        = string
  default     = "ray-llm-cluster"
}

variable "gpu_node_instance_type" {
  description = "Instance type for GPU nodes"
  type        = string
  default     = "g4dn.xlarge"
}

variable "gpu_node_min_size" {
  description = "Minimum number of GPU nodes"
  type        = number
  default     = 1
}

variable "gpu_node_max_size" {
  description = "Maximum number of GPU nodes"
  type        = number
  default     = 10
}

variable "gpu_node_desired_size" {
  description = "Desired number of GPU nodes"
  type        = number
  default     = 2
}

variable "cpu_node_instance_type" {
  description = "Instance type for CPU nodes"
  type        = string
  default     = "m5.2xlarge"
}

variable "cpu_node_min_size" {
  description = "Minimum number of CPU nodes"
  type        = number
  default     = 2
}

variable "cpu_node_max_size" {
  description = "Maximum number of CPU nodes"
  type        = number
  default     = 5
}

variable "cpu_node_desired_size" {
  description = "Desired number of CPU nodes"
  type        = number
  default     = 2
} 