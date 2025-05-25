import pulumi
import pulumi_gcp as gcp
import pulumi_docker as docker


# Configuration
config = pulumi.Config()
project_id = config.get("project-id") or "krishai-455907"
region = config.get("region") or "us-central1"
app_name = config.get("app-name") or "streamlit-app"

# Enable required APIs
apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com"
]

for api in apis:
    gcp.projects.Service(f"enable-{api.replace('.', '-')}", 
                        service=api,
                        project=project_id)

# Create Artifact Registry repository
registry = gcp.artifactregistry.Repository("streamlit-registry",
    location=region,
    repository_id=f"{app_name}-repo",
    description="Repository for Streamlit app images",
    format="DOCKER")

# Build and push Docker image
image = docker.Image("streamlit-image",
    image_name=pulumi.Output.concat(
        registry.location, "-docker.pkg.dev/",
        project_id, "/", registry.repository_id, "/", app_name, ":latest"
    ),
    build=docker.DockerBuildArgs(
        context=".",  # Path to your Streamlit app directory
        dockerfile="Dockerfile",
        platform="linux/amd64"  # Specify platform for Cloud Run
    ),
    registry=docker.RegistryArgs(
        username="oauth2accesstoken",
        password=pulumi.Output.secret(
            gcp.organizations.get_client_config().access_token
        )
    ))

# Create Cloud Run service
service = gcp.cloudrun.Service("streamlit-service",
    location=region,
    name=app_name,
    template=gcp.cloudrun.ServiceTemplateArgs(
        spec=gcp.cloudrun.ServiceTemplateSpecArgs(
            containers=[gcp.cloudrun.ServiceTemplateSpecContainerArgs(
                image=image.image_name,
                ports=[gcp.cloudrun.ServiceTemplateSpecContainerPortArgs(
                    container_port=8080
                )],
                resources=gcp.cloudrun.ServiceTemplateSpecContainerResourcesArgs(
                    limits={
                        "cpu": "2000m",
                        "memory": "2Gi"
                    }
                ),
                # Environment variables (add as needed)
                envs=[
                    # Example:
                    gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                         name="UPSTASH_VECTOR_REST_URL",
                         value="https://"
                     ),
                    gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                        name="UPSTASH_VECTOR_REST_TOKEN",
                        value=""
                    ),
                    gcp.cloudrun.ServiceTemplateSpecContainerEnvArgs(
                        name="OPENAI_API_KEY",
                        value="" # Listen on all interfaces
                    )
                ]
            )],
            container_concurrency=80,
            timeout_seconds=300
        ),
        metadata=gcp.cloudrun.ServiceTemplateMetadataArgs(
            annotations={
                "autoscaling.knative.dev/minScale": "0",
                "autoscaling.knative.dev/maxScale": "2",
                "run.googleapis.com/cpu-throttling": "false"
            }
        )
    ),
    traffics=[gcp.cloudrun.ServiceTrafficArgs(
        percent=100,
        latest_revision=True
    )])

# Make the service publicly accessible
iam_member = gcp.cloudrun.IamMember("streamlit-public-access",
    location=service.location,
    project=service.project,
    service=service.name,
    role="roles/run.invoker",
    member="allUsers")


# Optional: Custom domain mapping
# Uncomment and configure if you have a custom domain
# domain_mapping = gcp.cloudrun.DomainMapping("streamlit-domain",
#     location=region,
#     name="your-domain.com",
#     spec=gcp.cloudrun.DomainMappingSpecArgs(
#         route_name=service.name
#     ))

# Outputs
pulumi.export("service_url", service.statuses[0].url)
pulumi.export("service_name", service.name)
pulumi.export("registry_url", pulumi.Output.concat(
    registry.location, "-docker.pkg.dev/", project_id, "/", registry.repository_id
))