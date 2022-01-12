pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker {
                            label 'docker'
                            image 'dhealth/dev-pyeddl-base-cpu:439c6ea1'
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
                            image 'dhealth/dev-pyeddl-base-gpu:439c6ea1'
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
				sh 'wget -q https://www.dropbox.com/s/wap282xox5ew02d/cifar_trX.bin'
				sh 'wget -q https://www.dropbox.com/s/yxhw99cu1ktiwxq/cifar_trY.bin'
				sh 'wget -q https://www.dropbox.com/s/dh9vqxe9vt7scrp/cifar_tsX.bin'
				sh 'wget -q https://www.dropbox.com/s/gdmsve6mbu82ndp/cifar_tsY.bin'
				sh 'wget -q https://www.dropbox.com/s/re7jodd12srksd7/resnet18.onnx'
				sh 'bash examples/onnx/run_all_fast.sh'
				sh 'rm -fv mnist_*.bin'
				sh 'bash examples/NN/2_CIFAR10/run_all_fast.sh'
				sh 'wget -q https://www.dropbox.com/s/4m0h8ep53mixq6x/imdb_2000_trX.bin'
				sh 'wget -q https://www.dropbox.com/s/zekpjclm58tdevk/imdb_2000_trY.bin'
				sh 'wget -q https://www.dropbox.com/s/1bgdr8mz1lqkhgi/imdb_2000_tsX.bin'
				sh 'wget -q https://www.dropbox.com/s/6cwob77654lruwq/imdb_2000_tsY.bin'
				sh 'wget -q https://www.dropbox.com/s/2w0p7f4un6ci94v/eutrans_trX.bin'
				sh 'wget -q https://www.dropbox.com/s/g4k1bc6p4bow9tf/eutrans_trY.bin'
				sh 'wget -q https://www.dropbox.com/s/egcfin16gl9t92y/eutrans_tsX.bin'
				sh 'wget -q https://www.dropbox.com/s/n8ks3lyqyhxx1e8/eutrans_tsY.bin'
				sh 'wget -q https://www.dropbox.com/s/452pyxe9x5jpnwb/flickr_trX.bin'
				sh 'wget -q https://www.dropbox.com/s/24c2d5bm6pug8gg/flickr_trY.bin'
				sh 'bash examples/NN/4_NLP/run_all_fast.sh'
				sh 'rm -fv cifar_*.bin'
				sh 'rm -fv imdb_2000_*.bin eutrans_*.bin flickr_*.bin'
				sh 'rm -fv resnet18.onnx'
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
