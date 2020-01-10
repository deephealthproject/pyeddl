pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker {
                            label 'docker'
                            image 'simleo/eddl:c023a6e'
                            args '-u root:root'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
				echo 'Building'
				sh 'apt-get -y update && apt-get -y install --no-install-recommends python3-dev python3-pip'
				sh 'python3 -m pip install --upgrade --no-cache-dir setuptools pip numpy pybind11 pytest'
				sh 'python3 setup.py install'
                            }
                        }
                        stage('Test') {
                            steps {
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
                }
                stage('linux_gpu') {
                    agent {
                        docker {
                            label 'docker && gpu'
                            image 'simleo/eddl-gpu:c023a6e'
                            args '--gpus 1 -u root:root'
                        }
                    }
                    stages {
                        stage('Build') {
			    environment {
				EDDL_WITH_CUDA = 'true'
			    }
                            steps {
				echo 'Building'
				sh 'apt-get -y update && apt-get -y install --no-install-recommends python3-dev python3-pip'
				sh 'python3 -m pip install --upgrade --no-cache-dir setuptools pip numpy pybind11 pytest'
				sh 'ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcudart.so /usr/local/lib/'
				sh 'ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcurand.so /usr/local/lib/'
				sh 'ln -s /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcublas.so /usr/local/lib/'
				sh 'python3 setup.py install'
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
                }
            }
        }
    }
}
