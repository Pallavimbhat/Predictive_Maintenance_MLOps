pipeline {
    agent any

    stages {
        stage('Test') {
            steps {
                bat 'echo Jenkins Connected Successfully'
            }
        }
    }
}