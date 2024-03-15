from setuptools import find_packages, setup

package_name = 'create3_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='vscode',
    maintainer_email='andrewgerstenslager@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cliff_sensors_listener = create3_package.cliff_sensors:main',
            'pose_listener = create3_package.odom_sensor:main',
            'bump_listener = create3_package.bump_sensor:main',
            'scan_listener = create3_package.scan_sensor:main'
        ],
    },
)
