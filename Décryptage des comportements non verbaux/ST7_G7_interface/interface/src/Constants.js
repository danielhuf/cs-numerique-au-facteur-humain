const prod = {
  url: {
   API_URL: 'https://projectst7-group7.azurewebsites.net',
  }
};

const dev = {
  url: {
   API_URL: 'http://127.0.0.1:5000'
  }
};
 export const config = process.env.NODE_ENV === 'development' ? dev : prod;