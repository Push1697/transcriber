pipeline {
  agent any
  environment {
    REGISTRY = credentials('docker-registry-creds') // username/password
    IMAGE_NAME = "video-transcriber"
    IMAGE_TAG = "${env.BUILD_NUMBER}"
  }
  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }
    stage('Set up Python') {
      steps {
        sh 'python -m venv .venv'
        sh '. .venv/bin/activate && pip install --upgrade pip'
      }
    }
    stage('Install deps') {
      steps {
        sh '. .venv/bin/activate && pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu'
        sh '. .venv/bin/activate && pip install -r requirements.txt'
      }
    }
    stage('Lint/Syntax') {
      steps {
        sh '. .venv/bin/activate && python -m compileall app.py transcribe.py transcribe_simple.py'
      }
    }
    stage('Docker Build') {
      steps {
        sh 'docker build -t $IMAGE_NAME:$IMAGE_TAG .'
      }
    }
    stage('Docker Push') {
      when {
        expression { return env.REGISTRY != null }
      }
      steps {
        sh 'echo $REGISTRY_PSW | docker login -u $REGISTRY_USR --password-stdin'
        sh 'docker tag $IMAGE_NAME:$IMAGE_TAG $REGISTRY_USR/$IMAGE_NAME:$IMAGE_TAG'
        sh 'docker push $REGISTRY_USR/$IMAGE_NAME:$IMAGE_TAG'
      }
    }
    stage('Update Manifest') {
      steps {
        script {
          withCredentials([usernamePassword(credentialsId: 'git-creds', usernameVariable: 'GIT_USERNAME', passwordVariable: 'GIT_PASSWORD')]) {
            sh '''
              git config user.email "jenkins@example.com"
              git config user.name "Jenkins CI"
              # Update image tag in deployment.yaml
              sed -i "s|image: .*|image: $REGISTRY_USR/$IMAGE_NAME:$IMAGE_TAG|g" k8s/deployment.yaml
              git add k8s/deployment.yaml
              git commit -m "chore: update image tag to $IMAGE_TAG [skip ci]"
              git push https://$GIT_USERNAME:$GIT_PASSWORD@github.com/$GIT_USERNAME/video_transcriber.git HEAD:main
            '''
          }
        }
      }
    }
  }
  post {
    always {
      cleanWs()
    }
  }
}
