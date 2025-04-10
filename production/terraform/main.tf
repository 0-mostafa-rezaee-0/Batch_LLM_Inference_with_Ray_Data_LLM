provider "aws" {
  region = var.aws_region
}

module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  version = "~> 3.0"

  name = "ray-llm-vpc"
  cidr = "10.0.0.0/16"
  
  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false
  one_nat_gateway_per_az = true

  tags = {
    Environment = var.environment
    Project     = "ray-llm"
    Terraform   = "true"
  }
}

# EKS Cluster for Ray
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  version = "~> 18.0"

  cluster_name = "ray-llm-cluster"
  cluster_version = "1.24"
  
  vpc_id = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Node Groups for CPU workloads
  eks_managed_node_groups = {
    cpu = {
      name = "ray-cpu-nodes"
      instance_types = ["m5.2xlarge"]
      min_size = 2
      max_size = 5
      desired_size = 2
    }
  }

  # Node Groups for GPU workloads
  node_groups = {
    gpu = {
      name = "ray-gpu-nodes"
      instance_types = ["g4dn.xlarge"]
      min_size = 1
      max_size = 10
      desired_size = 2
      
      k8s_labels = {
        "accelerator" = "nvidia"
      }
      
      additional_tags = {
        "k8s.io/cluster-autoscaler/enabled" = "true"
        "k8s.io/cluster-autoscaler/ray-llm-cluster" = "owned"
      }
    }
  }

  tags = {
    Environment = var.environment
    Project     = "ray-llm"
    Terraform   = "true"
  }
}

# S3 bucket for model weights and data storage
resource "aws_s3_bucket" "model_storage" {
  bucket = "ray-llm-models-${var.environment}"
  
  tags = {
    Environment = var.environment
    Project     = "ray-llm"
    Terraform   = "true"
  }
}

# IAM policy for Ray to access S3
resource "aws_iam_policy" "ray_s3_access" {
  name        = "ray-s3-access-policy"
  description = "Policy for Ray to access S3 buckets"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
        ]
        Effect   = "Allow"
        Resource = [
          "${aws_s3_bucket.model_storage.arn}",
          "${aws_s3_bucket.model_storage.arn}/*",
        ]
      },
    ]
  })
}

# Output values
output "eks_cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "eks_cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "model_storage_bucket" {
  description = "S3 bucket for model storage"
  value       = aws_s3_bucket.model_storage.bucket
} 