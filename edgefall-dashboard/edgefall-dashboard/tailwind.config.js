/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        edge: {
          bg: '#f5f5f7',
          border: '#e5e7eb',
          red: '#e74c3c',
          redSoft: '#fdecea',
        },
      },
    },
  },
  plugins: [],
}
