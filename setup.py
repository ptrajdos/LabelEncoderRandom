from setuptools import setup, find_packages



setup(
        name='label_encoder_random',
        version ='0.0.1',
        author='Pawel Trajdos',
        author_email='pawel.trajdos@pwr.edu.pl',
        url = 'https://github.com/ptrajdos/LabelEncoderRandom',
        description="Label Encoder with random encoding",
        packages=find_packages(include=[
                'label_encoder_random',
                'label_encoder_random.*',
                ]),
        install_requires=[ 
                'numpy>=1.22.4',
                'joblib',
                'scikit-learn>=1.2.2',
        ],
        test_suite='test'
        )
