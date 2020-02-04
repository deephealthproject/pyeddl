pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker {
                            label 'docker'
                            image 'simleo/pyeddl-base:c023a6e'
                            args '-u 1000:1000'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
				echo 'Building'
				sh 'python3 setup.py install --user'
                            }
                        }
                        stage('Test') {
                            steps {
				echo 'Downloading test dataset'
				sh 'wget -q  https://www.dropbox.com/s/khrb3th2z6owd9t/trX.bin'
				sh 'wget -q https://www.dropbox.com/s/m82hmmrg46kcugp/trY.bin'
				sh 'wget -q https://www.dropbox.com/s/7psutd4m4wna2d5/tsX.bin'
				sh 'wget -q https://www.dropbox.com/s/q0tnbjvaenb4tjs/tsY.bin'
				echo 'Testing'
				sh 'pytest tests'
				sh 'python3 examples/Tensor/eddl_tensor.py'
				sh 'python3 examples/NN/other/eddl_ae.py --epochs 1'
                            }
                        }
                        stage('linux_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                    post {
                        cleanup {
                            deleteDir()
                        }
                    }
                }
                stage('linux_gpu') {
                    agent {
                        docker {
                            label 'docker && gpu'
                            image 'simleo/pyeddl-gpu-base:c023a6e'
                            args '--gpus 1 -u 1000:1000'
                        }
                    }
                    stages {
                        stage('Build') {
			    environment {
				EDDL_WITH_CUDA = 'true'
			    }
                            steps {
				echo 'Building'
				sh 'python3 setup.py install --user'
			    }
                        }
                        stage('Test') {
                            steps {
				echo 'Testing'
				sh 'pytest tests'
				sh 'python3 examples/Tensor/eddl_tensor.py --gpu'
				sh 'bash examples/NN/other/run_all_fast.sh'
				sh 'bash examples/NN/1_MNIST/run_all_fast.sh'
			    }
                        }
                        stage('linux_gpu_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                    post {
                        cleanup {
                            deleteDir()
                        }
                    }
                }
            }
        }
    }
}
