$(document).ready(function() {
  // Dark mode switch
  $( ".js--dark-mode-switch" ).click(function() {
    // Darkens the switch
    $(this).toggleClass('darkened-switch');

    // Darkens the home button
    $('.js--home-button').toggleClass('darkened-home-button');

    // Darkens the headers
    $('h1').toggleClass('darkened-header');
    $('h2').toggleClass('darkened-header');
    $('h3').toggleClass('darkened-header');

    // Darkens the button
    $('.button').toggleClass('darkened-button');

    // Darkens the contents
    $('article a').toggleClass('darkened-link');

    // Darkens the background
    $('body').toggleClass('darkened-background');
    $('html').toggleClass('darkened-background');
  });
});