pipeline {
    agent any

    stages {

        stage('Clone Repository') {
            steps {
                git branch: 'main',
                url: 'https://github.com/Pallavimbhat/Predictive_Maintenance_MLOps.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t predictive-maintenance .'
            }
        }

        stage('Run Docker Container') {
            steps {
                bat 'docker run -d -p 5001:10000 predictive-maintenance'
            }
        }
    }
}