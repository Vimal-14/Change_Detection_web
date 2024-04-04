$(document).ready(function() {
  // Sticky Navigation
  $('.js--section-about').waypoint(
    function(direction) {
      if (direction == 'down') {
        $('nav').addClass('sticky');
      } else {
        $('nav').removeClass('sticky');
      }
    },
    {
      offset: '60px'
    }
  );

  // Scroll animation
  $('.js--features').waypoint(
    function(direction) {
      $('.js--features').addClass('animated fadeIn');
    },
    {
      offset: '50%'
    }
  );

  // Mobile navigation
  $('.js--mobile-navigation-icon').click(function() {
    var navigationLinks = $('.js--navigation-links');
    var mobileNavigatioIcon = $('.js--mobile-navigation-icon ion-icon');

    navigationLinks.slideToggle(200);

    if (mobileNavigatioIcon.attr('name') === 'menu') {
      mobileNavigatioIcon.removeAttr('name');
      mobileNavigatioIcon.attr('name', 'close');
    } else if (mobileNavigatioIcon.attr('name') === 'close') {
      mobileNavigatioIcon.removeAttr('name');
      mobileNavigatioIcon.attr('name', 'menu');
    }
  });
});