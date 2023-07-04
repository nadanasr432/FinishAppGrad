const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const resetPasswordSchema = new Schema({
  emailAddress: {
    type: String,
    required: true
  }
});

const ResetPassword = mongoose.model('ResetPassword', resetPasswordSchema);

module.exports = ResetPassword;