#include <iostream>
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

int main() {
    // Open the serial port
    int serialPort = open("/dev/ttyUSB0", O_RDWR | O_NOCTTY | O_NDELAY);
    if (serialPort == -1) {
        std::cerr << "Failed to open the serial port." << std::endl;
        return 1;
    }

    // Configure the serial port settings
    struct termios serialSettings;
    tcgetattr(serialPort, &serialSettings);
    cfsetispeed(&serialSettings, B9600);  // Set baud rate to 9600
    cfsetospeed(&serialSettings, B9600);
    serialSettings.c_cflag &= ~PARENB;    // Disable parity bit
    serialSettings.c_cflag &= ~CSTOPB;    // Set one stop bit
    serialSettings.c_cflag &= ~CSIZE;     // Clear data size bits
    serialSettings.c_cflag |= CS8;        // Set 8 data bits
    tcsetattr(serialPort, TCSANOW, &serialSettings);

    // Send command to the serial port
    std::string command = "AT\r\n";
    write(serialPort, command.c_str(), command.length());

    // Read response from the serial port
    char response[256];
    ssize_t bytesRead = read(serialPort, response, sizeof(response) - 1);
    if (bytesRead > 0) {
        response[bytesRead] = '\0';
        std::cout << "Received response: " << response << std::endl;
    }

    // Close the serial port
    close(serialPort);

    return 0;
}
