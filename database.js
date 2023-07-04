
const mongoose = require('mongoose');
mongoose.connect('mongodb+srv://nad:mmmmmnnnnnaaaaa@atlascluster.lf1llpa.mongodb.net/?retryWrites=true&w=majority'
, { useNewUrlParser: true, useUnifiedTopology: true })
.then(() => console.log('Connected to MongoDB'))
.catch((err) => console.error('Failed to connect to MongoDB', err));
const userSchema = new mongoose.Schema({
  fullName:String,
  emailAddress: String,
  password: String,
  phoneNumber: String,
  age: Number,
  date: Date,
  chronicDisease: String,
  gender: String,
  imagePath: String,
});

const User = mongoose.model('User', userSchema);
module.exports = User;
