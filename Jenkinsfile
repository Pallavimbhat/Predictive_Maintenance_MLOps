pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git 'https://github.com/Pallavimbhat/Predictive_Maintenance_MLOps.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t predictive-maintenance .'
            }
        }

        stage('Run Docker Container') {
            steps {
                bat 'docker run -d -p 5000:8080 predictive-maintenance'
            }
        }
    }
}