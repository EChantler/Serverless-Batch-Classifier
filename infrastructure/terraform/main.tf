provider "aws" {
  region = var.aws_region
}
 
# resource "aws_rds_cluster" "aurora_serverless" {
#   cluster_identifier      = "aurora-serverless-cluster"
#   engine                  = "aurora-mysql"
#   # Remove engine_mode for Aurora Serverless v2
#   master_username         = var.aurora_master_username
#   master_password         = var.aurora_master_password
#   backup_retention_period = 1
#   skip_final_snapshot     = true

#   # Aurora Serverless V2 requires `serverlessv2_scaling_configuration`
#   serverlessv2_scaling_configuration {
#     min_capacity = var.aurora_min_capacity
#     max_capacity = var.aurora_max_capacity
#   }
# }
# Add security group for DB access
data "aws_vpc" "default" {
  default = true
}

resource "aws_security_group" "db_sg" {
  name        = "db-security-group"
  description = "Allow MySQL access from anywhere"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 3306
    to_port     = 3306
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow MySQL access"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }
}

resource "aws_db_instance" "db_instance" {
  identifier             = "batch-classifier-db"
  engine                 = "mysql"
  engine_version         = "8.0"
  instance_class         = "db.t3.micro"
  allocated_storage      = 20
  db_name                = "batchdb"
  username               = var.aurora_master_username
  password               = var.aurora_master_password
  publicly_accessible    = true
  vpc_security_group_ids = [aws_security_group.db_sg.id]
  skip_final_snapshot    = true
}

resource "aws_s3_bucket" "lambda_bucket" {
  bucket = var.s3_bucket_name
}

resource "aws_iam_role" "lambda_exec_role" {
  name = "lambda_exec_role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "lambda_s3_policy" {
  name = "lambda_s3_policy"
  role = aws_iam_role.lambda_exec_role.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
      Resource = [
        aws_s3_bucket.lambda_bucket.arn,
        "${aws_s3_bucket.lambda_bucket.arn}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_exec_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}
resource "aws_lambda_function" "main_lambda" {
  function_name = "main_lambda"
  package_type  = "Image"
  image_uri     = var.lambda_image_uri
  role          = aws_iam_role.lambda_exec_role.arn
  timeout       = 60
  
  # Specify architecture for container images
  architectures = ["x86_64"]
  
  # Add environment variables if needed
  environment {
    variables = {
      LOG_LEVEL = "INFO"
    }
  }
}

resource "aws_api_gateway_rest_api" "api" {
  name        = "ServerlessBatchClassifierAPI"
  description = "API Gateway to invoke Lambda"
}

resource "aws_api_gateway_resource" "lambda_resource" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  parent_id   = aws_api_gateway_rest_api.api.root_resource_id
  path_part   = "invoke"
}

resource "aws_api_gateway_method" "invoke_method" {
  rest_api_id   = aws_api_gateway_rest_api.api.id
  resource_id   = aws_api_gateway_resource.lambda_resource.id
  http_method   = "POST"
  authorization = "NONE"
}

resource "aws_api_gateway_integration" "lambda_integration" {
  rest_api_id = aws_api_gateway_rest_api.api.id
  resource_id = aws_api_gateway_resource.lambda_resource.id
  http_method = aws_api_gateway_method.invoke_method.http_method
  integration_http_method = "POST"
  type                    = "AWS_PROXY"
  uri                     = aws_lambda_function.main_lambda.invoke_arn
}

resource "aws_cloudwatch_event_rule" "invoke_lambda_rule" {
  name        = "InvokeLambdaRule"
  description = "EventBridge rule to invoke Lambda"
  event_pattern = <<EOF
{
  "source": ["custom.source"]
}
EOF
}

resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.invoke_lambda_rule.name
  arn       = aws_lambda_function.main_lambda.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.main_lambda.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.invoke_lambda_rule.arn
}

resource "aws_lambda_permission" "allow_apigw" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.main_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = aws_api_gateway_rest_api.api.execution_arn
}
