DOCKERFILE=Dockerfile
NEW_DOCKERIMAGE=alchemz/stablediffusion:torch:1.13.0_v1
CUDA_VERSION="11.8.0"
BASE_DOCKER_IMAGE=nvcr.io/nvidia/pytorch:22.09-py3

cd $WORK_DIR

buildah bud \
        --build-arg BASE_IMAGE=${BASE_DOCKER_IMAGE} \
        -t ${NEW_DOCKERIMAGE} ${DOCKERFILE}

buildah push ${NEW_DOCKERIMAGE}
