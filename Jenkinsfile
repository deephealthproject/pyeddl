pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker {
                            label 'docker'
                            image 'simleo/pyeddl-base:02e37c0d'
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
				sh 'wget -q https://www.dropbox.com/s/khrb3th2z6owd9t/mnist_trX.bin'
				sh 'wget -q https://www.dropbox.com/s/m82hmmrg46kcugp/mnist_trY.bin'
				sh 'wget -q https://www.dropbox.com/s/7psutd4m4wna2d5/mnist_tsX.bin'
				sh 'wget -q https://www.dropbox.com/s/q0tnbjvaenb4tjs/mnist_tsY.bin'
				echo 'Testing'
				sh 'pytest tests'
				sh 'python3 examples/Tensor/array_tensor_save.py'
				sh 'python3 examples/NN/1_MNIST/mnist_auto_encoder.py --epochs 1'
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
                            image 'simleo/pyeddl-gpu-base:02e37c0d'
                            args '--gpus 1'
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
				sh 'python3 examples/Tensor/array_tensor_save.py'
				sh 'wget -q https://www.dropbox.com/s/khrb3th2z6owd9t/mnist_trX.bin'
				sh 'wget -q https://www.dropbox.com/s/m82hmmrg46kcugp/mnist_trY.bin'
				sh 'wget -q https://www.dropbox.com/s/7psutd4m4wna2d5/mnist_tsX.bin'
				sh 'wget -q https://www.dropbox.com/s/q0tnbjvaenb4tjs/mnist_tsY.bin'
				sh 'bash examples/NN/1_MNIST/run_all_fast.sh'
				sh 'bash examples/NN/py_loss_metric/run_all_fast.sh'
				sh 'bash examples/onnx/run_all_fast.sh'
				sh 'rm -fv mnist_*.bin'
				sh 'wget -q https://www.dropbox.com/s/wap282xox5ew02d/cifar_trX.bin'
				sh 'wget -q https://www.dropbox.com/s/yxhw99cu1ktiwxq/cifar_trY.bin'
				sh 'wget -q https://www.dropbox.com/s/dh9vqxe9vt7scrp/cifar_tsX.bin'
				sh 'wget -q https://www.dropbox.com/s/gdmsve6mbu82ndp/cifar_tsY.bin'
				sh 'bash examples/NN/2_CIFAR10/run_all_fast.sh'
				sh 'rm -fv cifar_*.bin'
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
