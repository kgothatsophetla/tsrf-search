/** @type {import('next').NextConfig} */
const config = {
  async rewrites() {
    if (process.env.NODE_ENV !== "development") {
      return [];
    }
    const apiOrigin = process.env.API_ORIGIN ?? "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${apiOrigin}/:path*`,
      },
    ];
  },
};

export default config;
