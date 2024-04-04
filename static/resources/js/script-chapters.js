$(document).ready(function() {
  // Dark mode switch
  $( ".js--dark-mode-switch" ).click(function() {
    // Darkens the switch
    $(this).toggleClass('darkened-switch');

    // Darkens the home button
    $('.js--home-button').toggleClass('darkened-home-button');

    // Darkens the text
    $('main').toggleClass('darkened-text');

    // Darkens the background
    $('body').toggleClass('darkened-background');
    $('html').toggleClass('darkened-background');
  });
});