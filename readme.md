### VNN 🪢
web application for visualizing neural networks in real time

[![Next.js](https://img.shields.io/badge/Next.js-12.0.0-blue)](https://nextjs.org/)
[![License](https://img.shields.io/github/license/dinxsh/vnn)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](https://github.com/dinxsh/vnn/pulls)
[![Stage](https://img.shields.io/badge/stage-beta-yellow)](https://github.com/dinxsh/vnn)

![image](https://github.com/user-attachments/assets/987915c2-9da2-4602-b175-480ed4f4a9be)

### installation

1. **clone the repository:**
   ```bash
   git clone https://github.com/dinxsh/vnn.git
   cd vnn
   ```

2. **setup backend:**
   - ensure [go](https://golang.org/doc/install) is installed.
   - navigate to the backend directory:
     
     ```bash
     cd backend
     ```
   - build and run the server:
     ```bash
     go build
     ./backend
     ```

3. **setup frontend:**
   - ensure [node.js](https://nodejs.org/en/download/) is installed.
   - navigate to the frontend directory:
     
     ```bash
     cd ../frontend
     ```
   - install dependencies and run the development server:
     
     ```bash
     npm install
     npm run dev
     ```

## usage

- try at `http://localhost:3000`.
- load and visualize neural network models.

## license

this project is licensed under the MIT License. For details, see the [LICENSE](LICENSE) file.
